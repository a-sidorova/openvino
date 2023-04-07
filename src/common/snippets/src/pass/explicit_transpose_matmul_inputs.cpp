// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/op/subgraph.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>



ngraph::snippets::pass::ExplicitTransposeMatMulInputs::ExplicitTransposeMatMulInputs() {
    MATCHER_SCOPE(ExplicitTransposeMatMulInputs);

    auto m_matmul0 = std::make_shared<ngraph::opset1::MatMul>(
            ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
            ngraph::pattern::any_input(ngraph::pattern::has_static_shape()));

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(m_matmul0, matcher_name),
        [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ExplicitTransposeMatMulInputs")
        auto root = m.get_match_root();
        bool rewritten = false;

        auto matmul = ngraph::as_type_ptr<ngraph::opset1::MatMul>(root);
        if (!matmul)
            return false;

        for (size_t i = 0; i < matmul->get_input_size(); i++) {
            if (i == 0 && !matmul->get_transpose_a())
                continue;
            if (i == 1 && !matmul->get_transpose_b())
                continue;

            auto parent = matmul->get_input_node_shared_ptr(i);
            auto transpose = ngraph::as_type_ptr<ngraph::opset1::Transpose>(parent);
            while (!transpose && !ov::is_type<ngraph::opset1::Parameter>(parent)) {
                // We can set supported order and transposed_b(false) only if ops have scalar shapes to avoid shape mismatching
                const auto parent_count = parent->inputs().size();
                bool are_weights_scalar = true;
                for (size_t j = 1; j < parent_count; ++j) {
                    are_weights_scalar = are_weights_scalar && ngraph::shape_size(parent->get_input_shape(j)) == 1;
                }
                if (!are_weights_scalar)
                    break;

                parent = parent->get_input_node_shared_ptr(0);
                transpose = ngraph::as_type_ptr<ngraph::opset1::Transpose>(parent);
            }
            // If there isn't another Transpose, need to create new Transpose
            if (transpose) {
                const auto transpose_pattern = ngraph::as_type_ptr<ngraph::opset1::Constant>(transpose->get_input_node_shared_ptr(1));
                if (!transpose_pattern)
                    continue;

                auto transposed_order = transpose_pattern->cast_vector<int32_t>();
                std::swap(*transposed_order.rbegin(), *(transposed_order.rbegin() + 1));

                auto new_transpose_order = std::make_shared<ngraph::opset1::Constant>(transpose_pattern->get_element_type(),
                                                                                      ngraph::Shape{4},
                                                                                      transposed_order);
                new_transpose_order->set_friendly_name(transpose_pattern->get_friendly_name());
                ngraph::copy_runtime_info(transpose_pattern, new_transpose_order);
                transpose->set_argument(1, new_transpose_order);
            } else {  // Otherwise, merge existing and new Transpose orders
                OPENVINO_ASSERT(ov::is_type<opset1::Parameter>(parent),
                                "ExplicitTransposeMatMulInputs expects Parameter in cases when there isn't existing Transpose on input");
                const auto& consumers = parent->get_output_target_inputs(0);
                OPENVINO_ASSERT(consumers.size() == 1,
                                "ExplicitTransposeMatMulInputs expects Parameter with one consumer in cases when there isn't existing Transpose on input");
                // Extract Transpose from MatMul
                const auto rank = matmul->get_input_shape(1).size();
                std::vector<size_t> transpose_order(rank, 0);
                std::iota(transpose_order.begin(), transpose_order.end(), 0);
                std::swap(transpose_order[rank - 1], transpose_order[rank - 2]);

                const auto constant_order = std::make_shared<opset1::Constant>(ov::element::i32, ov::Shape{rank}, transpose_order);
                const auto new_transpose = std::make_shared<opset1::Transpose>(parent, constant_order); // parent is Parameter
                const auto consumer_input = *(consumers.begin());
                consumer_input.replace_source_output(new_transpose);
            }

            if (i == 0) {
                matmul->set_transpose_a(false);
            } else {
                matmul->set_transpose_b(false);
            }
            rewritten |= true;
        }

        return rewritten;
    });
}
