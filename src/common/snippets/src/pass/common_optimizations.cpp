// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/common_optimizations.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "transformations/utils/utils.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/softmax_reshape_elimination.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"
#include "openvino/core/parallel.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::CommonOptimizations, "Snippets::CommonOptimizations");

namespace ngraph {
namespace snippets {
namespace pass {

namespace {

std::vector<size_t> get_factors(size_t dim) {
    std::vector<size_t> factors;
    size_t div = 2;
    while (div <= dim) {
        const auto res = dim / div;
        if (res * div == dim) {
            factors.push_back(div);
            dim = res;
        } else {
            div++;
        }
    }
    return factors;
}

size_t get_lcm(size_t lhs, size_t rhs) {
    size_t lcm = lhs * rhs;
    while (lhs != 0 && rhs != 0) {
        if (lhs > rhs)
            lhs %= rhs;
        else
            rhs %= lhs;
    }
    return lcm / (lhs + rhs);
}
} // namespace


void CommonOptimizations::ExtractConstants(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ExtractConstants");
    auto body = subgraph->body_ptr();

    ParameterVector new_parameters;
    OutputVector new_external_inputs = subgraph->input_values();

    for (auto& op : body->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || ngraph::shape_size(constant->get_shape()) == 1ul)
            continue;

        const auto child = constant->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        if (op::Subgraph::constant_input_should_be_inside_body(child))
            continue;

        auto parameter = std::make_shared<opset1::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());
        parameter->set_friendly_name(constant->get_friendly_name());
        ngraph::copy_runtime_info(constant, parameter);
        constant->output(0).replace(parameter->output(0));

        new_external_inputs.push_back(constant);
        new_parameters.push_back(parameter);
    }

    if (new_parameters.size() != 0) {
        body->add_parameters(new_parameters);
        body->validate_nodes_and_infer_types();
        subgraph->set_arguments(new_external_inputs);
    }
}

bool CommonOptimizations::ExtractUnsupportedTransposes(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ExtractUnsupportedTransposes");
    const auto& body = subgraph->body_ptr();
    const auto parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs (it's possible case in general),
    //           we cannot garantee correct extraction since we don't have correct connestions between body I/O and Subgraph I/O.
    if (parameters.size() != subgraph->input_values().size())
        return false;

    bool updated = false;
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& parameter = parameters[i];
        const auto& consumers = parameter->get_output_target_inputs(0);
        if (consumers.size() != 1)
            continue;

        const auto transpose = ov::as_type_ptr<opset1::Transpose>(consumers.begin()->get_node()->shared_from_this());
        if (!transpose)
            continue;

        const auto& order = ov::as_type_ptr<opset1::Constant>(transpose->get_input_node_shared_ptr(1));
        if (!order)
            continue;

        const auto order_value = order->cast_vector<int>();
        const auto transpose_child = *(transpose->get_output_target_inputs(0).begin());
        const auto is_brgemm_case = ov::is_type<opset1::MatMul>(transpose_child.get_node()->shared_from_this());
        if ((is_brgemm_case && FuseTransposeBrgemm::supported_cases.count(order_value) != 0) ||
            (TransposeDecomposition::supported_cases.count(order_value) != 0))
            continue;

        // If the transpose isn't supported - we have to extract from Subgraph
        transpose->set_argument(0, subgraph->input_value(i));
        subgraph->set_argument(i, transpose);
        transpose_child.replace_source_output(parameter);
        // Update shape
        parameter->set_partial_shape(transpose->get_output_partial_shape(0));
        updated = true;
    }

    return updated;
}

bool CommonOptimizations::canBeParallelOptimized(const std::shared_ptr<const ov::Node>& node) {
    if (ov::is_type<ov::op::v0::MatMul>(node)) {
        // It's needed only for 3D MHA patterns
        const auto mm_shape = node->get_shape();
        if (mm_shape.size() != 3)
            return false;

        const auto current_parallel_work_amount =
            std::accumulate(mm_shape.rbegin() + 2, mm_shape.rend(), 1lu, std::multiplies<size_t>());
        const auto dim_M = *(mm_shape.rbegin() + 1);
        const auto nthrs = static_cast<size_t>(parallel_get_num_threads());
        return (current_parallel_work_amount < nthrs) && (current_parallel_work_amount * dim_M >= nthrs);
    } else {
        // TODO: Add common support
        return false;
    }
}

bool CommonOptimizations::SplitDimensions(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph) {
    // To increase parallelism work in 3D cases for MHA pattern,
    // we split 1st dimension (starting from 0th) into 2 new dimensions to get 4D Shapes where
    // - 0th and 1st dimensions are used in parallel scheduling,
    // - 2nd and 3rd dimensions are used in kernel
    // Note: 3D Patterns don't contain Transpose inside so the reshaping is valid

    // It's needed only for MHA patterns. Need to add support for common patterns
    if (!subgraph->has_domain_sensitive_ops())
        return false;

    const auto& body = subgraph->body_ptr();
    const auto& parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs (it's possible case in general),
    //           we cannot garantee correct extraction since we don't have correct connestions between body I/O and Subgraph I/O.
    if (parameters.size() != subgraph->input_values().size())
        return false;

    // Need to find MatMul0 and check output shape
    const auto& ops = body->get_ordered_ops();
    const auto mm_it = std::find_if(ops.begin(), ops.end(),
                                    [](const std::shared_ptr<ov::Node>& node){ return ov::is_type<ov::op::v0::MatMul>(node); });
    const auto transpose_it = std::find_if(ops.begin(), ops.end(),
                                    [](const std::shared_ptr<ov::Node>& node){ return ov::is_type<ov::op::v1::Transpose>(node); });
    // The optimization doesn't support body with Transpose because
    // Snippets support Transpose with certain order
    if (mm_it == ops.end() || transpose_it != ops.end())
        return false;

    const auto matmul0 = ov::as_type_ptr<ov::op::v0::MatMul>(*mm_it);
    if (!matmul0 || !canBeParallelOptimized(matmul0))
        return false;

    auto get_dim_M = [](const ov::Shape& shape) {
        return *(shape.rbegin() + 1);
    };

    const auto mm_shape = matmul0->get_shape();
    const auto optimal_work_amount = static_cast<size_t>(parallel_get_num_threads());
    const auto batch_dim =
        std::accumulate(mm_shape.rbegin() + 2, mm_shape.rend(), 1, std::multiplies<size_t>());  // b
    const auto m_dim = get_dim_M(mm_shape);  // m

    size_t batch_m_dim = 1;
    size_t new_m_dim = m_dim;

    // Need to find optimized dimension splitting.
    // The work amount for parallelism should be divided by max thread count in ideal case
    // that all threads have the same full work amount (avoid of thread downtime)
    // If it's impossible, it should be more than max thread count
    // TODO: Find solution for finding of optimal splitting in these cases
    // For example, there are 36 threads and shape [20, 288, 32]
    //              LCM(20, 36) = 180 <- ideal work amount for parallelism
    //              new_shape [20, 180 / 20, 288 / (180 / 20), 32 ] => [20, 9, 32, 32]
    //              Each thread has work_amount = 20 * 9 / nthrs = 5
    const auto lcm = get_lcm(batch_dim, optimal_work_amount);  // LCM(b, nthrs)
    const auto batch_dim_multiplier = lcm / batch_dim;  // LCM(b, nthrs) / b
    const auto needed_new_dim = m_dim / batch_dim_multiplier;  // m / (LCM(b, nthrs) / b) - needed factors of dimension m
    if (batch_dim_multiplier * needed_new_dim == m_dim) {
        batch_m_dim = batch_dim_multiplier;
        new_m_dim = needed_new_dim;
    } else {
        const auto m_factors = get_factors(m_dim);
        size_t idx = 0;
        while (batch_m_dim * batch_dim < optimal_work_amount && idx < m_factors.size()) {
            batch_m_dim *= m_factors[idx];
            idx++;
        }
        new_m_dim = m_dim / batch_m_dim;
    }

    OPENVINO_ASSERT(batch_m_dim * new_m_dim == m_dim, "Incorrect dimension M splitting!");

    bool updated = false;

    std::set<std::shared_ptr<ov::op::v0::Parameter>> reshaped_params;

    auto insert_reshape = [&](const std::shared_ptr<ov::op::v0::Parameter>& param, const ov::Shape& new_shape) {
        const auto index = std::distance(parameters.begin(), std::find(parameters.begin(), parameters.end(), param));
        const auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{new_shape.size()}, new_shape);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(subgraph->input_value(index), shape_const, false);
        subgraph->input(index).replace_source_output(reshape);
        param->set_partial_shape(new_shape);
        reshaped_params.insert(param);
        updated = true;
    };

    auto reshape_parameter = [&](const std::shared_ptr<ov::Node>& node, bool split_m_dim = true) {
        const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node);
        if (!param || reshaped_params.count(param) > 0)
            return;
        const auto shape = param->get_partial_shape().get_shape();
        ov::Shape new_shape = shape;
        if (split_m_dim) {
            const auto current_m_dim = get_dim_M(shape);
            OPENVINO_ASSERT(current_m_dim == 1 || current_m_dim == m_dim, "Incorrect shape for splitting!");
            if (current_m_dim == 1) {
                new_shape.insert((new_shape.rbegin() + 2).base(), 1);
            } else {
                new_shape.insert((new_shape.rbegin() + 2).base(), batch_m_dim);
                *(new_shape.rbegin() + 1) = new_m_dim;
            }
        } else {
            new_shape.insert((new_shape.rbegin() + 2).base(), 1);
        }
        OPENVINO_ASSERT(ov::shape_size(new_shape) == ov::shape_size(shape), "Incorrect shape splitting!");
        insert_reshape(param, new_shape);
    };

    /***** Reshape insertion *****/

    auto update_matmul_second_branch = [&](const std::shared_ptr<ov::Node>& node) {
        auto parent = node->get_input_node_shared_ptr(1);
        while (!ov::is_type<ov::op::v0::Parameter>(parent)) {
            if (parent->get_input_size() > 1) {
                for (const auto& input_source : parent->input_values()) {
                    reshape_parameter(input_source.get_node_shared_ptr(), false);
                }
            }

            // [107731]: It's covered my MHA tokenization
            parent = parent->get_input_node_shared_ptr(0);
        }
        reshape_parameter(parent, false);
    };

    for (const auto& op : ops) {
        if (ov::is_type<ov::op::v0::MatMul>(op)) {
            update_matmul_second_branch(op);
        }
    }

    // Update All M dimensions
    for (const auto& param : parameters) {
        if (reshaped_params.count(param) == 0)
            reshape_parameter(param, true);
    }

    // Return the previous shape on outputs
    for (size_t i = 0; i < subgraph->get_output_size() && updated; ++i) {
        const auto output_shape = subgraph->get_output_shape(i);
        if (is_scalar(output_shape))
            continue;

        const auto& target_inputs = subgraph->get_output_target_inputs(i);
        const auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{output_shape.size()}, output_shape);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(subgraph->output(i), shape_const, false);
        // Save output name
        const auto original_output = body->get_results()[i]->get_input_node_shared_ptr(0);
        const auto original_name = original_output->get_friendly_name();
        reshape->set_friendly_name(original_name);
        original_output->set_friendly_name(original_name + "_original");

        for (const auto& input : target_inputs) {
            input.replace_source_output(reshape);
        }
        updated = true;
    }

    // Need to update Softmax Axis
    if (updated) {
        for (const auto &op : ops) {
            if (const auto softmax_v8 = ngraph::as_type_ptr<ov::op::v8::Softmax>(op)) {
                softmax_v8->set_axis(-1);
            } else if (const auto softmax_v1 = ngraph::as_type_ptr<ov::op::v1::Softmax>(op)) {
                softmax_v1->set_axis(3);
            }
        }
    }

    return updated;
}

CommonOptimizations::CommonOptimizations() {
    MATCHER_SCOPE(CommonOptimizations);
    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::CommonOptimizations");

        auto subgraph = ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        const auto& body = subgraph->body_ptr();
        const auto is_quantized = subgraph->is_quantized();

        // Firsly we should transform all original Converts inside body to ConvertTruncation to save original behavior.
        // Then if Subgraph contains FakeQuantize we enable specific transformation for quantized subgraphs.
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::snippets::pass::TransformConvertToConvertTruncation>();
        manager.register_pass<ngraph::snippets::pass::ExplicitTransposeMatMulInputs>();
        if (is_quantized) {
            manager.register_pass<ngraph::snippets::pass::CommonFakeQuantizeDecomposition>();
        }
        manager.register_pass<snippets::pass::SoftmaxReshapeElimination>();
        manager.run_passes(body);

        // At the moment only non-scalar Constants of FakeQuantize can be inside Subgraph
        // so we can enable ExtractConstants pass for quantized models
        if (is_quantized) {
            ExtractConstants(subgraph);
        }
        // The passes only for MHA pattern
        if (subgraph->has_domain_sensitive_ops()) {
            bool need_validation = false;
            auto register_transformation = [&](const std::function<bool(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph)>& pass) -> void {
                if (pass(subgraph))
                    need_validation = true;
            };
            register_transformation(ExtractUnsupportedTransposes);
            register_transformation(SplitDimensions);
            if (need_validation) {
                subgraph->validate_and_infer_types();
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<ngraph::snippets::op::Subgraph>(),
                                                        matcher_name);
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
