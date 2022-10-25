// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>

#include <ngraph/pass/manager.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {

auto getRegisters(const std::shared_ptr<ngraph::Node> &n) -> RegInfo {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::getRegisters")

    // ToDo: change to reg_t
    std::vector<size_t> rin, rout;

    for (const auto& output : n->outputs()) {
        const auto& rt = output.get_tensor_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end())
            rout.push_back(it_rt->second.as<size_t>());
    }

    for (const auto& input : n->inputs()) {
        auto rt = input.get_source_output().get_tensor_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end())
            rin.push_back(it_rt->second.as<size_t>());
    }
    return std::make_pair(rin, rout);
}

ngraph::snippets::code ngraph::snippets::Generator::generate(std::shared_ptr<ov::Model>& m,
                                                             const void* compile_params) const {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code generation");

    auto params = m->get_parameters();
    auto results = m->get_results();
    auto in = params.size();
    auto out = results.size();

    std::vector<size_t> io_last_dims(in + out);
    std::vector<size_t> io_data_sizes(in + out);
    std::transform(params.begin(), params.end(), io_last_dims.begin(),
                   [](const std::shared_ptr<Node>& n){
                       auto last_dim = n->get_output_partial_shape(0).rbegin();
                       return last_dim->is_dynamic() ? op::Subgraph::DYNAMIC_DIMENSION
                                                     : last_dim->get_length();
                   });
    std::transform(results.begin(), results.end(), io_last_dims.begin() + in,
                   [](const std::shared_ptr<Node> &n) {
                       auto last_dim = n->get_input_partial_shape(0).rbegin();
                       return last_dim->is_dynamic() ? op::Subgraph::DYNAMIC_DIMENSION
                                                     : last_dim->get_length();
                   });
    std::transform(params.begin(), params.end(), io_data_sizes.begin(),
                   [](const std::shared_ptr<Node>& n){return n->get_element_type().size();});
    std::transform(results.begin(), results.end(), io_data_sizes.begin() + in,
                   [](const std::shared_ptr<Node>& n){return n->get_element_type().size();});

    OV_ITT_TASK_CHAIN(GENERATE, ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::VectorTile")
    // vector loop
    std::vector<AllocatedEmitter> lowered;
    auto lower_ops = [&lowered, this](const NodeVector& ops){
        std::transform(ops.begin(), ops.end(), std::back_inserter(lowered),
                       [this](const std::shared_ptr<Node>& n){
                           return std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n));
                       });
    };
    // *1* solo vector/scalar loop + empty outer loop
    //      => skip increments (both counter & ptr) : set evaluate_once flag
    // *2* solo vector/scalar loop + non-empty outer loop
    //      => skip counter increments but perform ptr increments : set evaluate_once,
    //         and perform pointer increments through finalization offsets
    // *3* one vector loop + multiple scalar loops
    //      => vector force_ptr_increment=true to enable *2*, scalar as usual
    // *4* vector loop(s) + one scalar loop
    //      => vector as usual, scalar depends on outer loop, see *1* and *2*
    auto optimize_single_evaluation = [](const std::shared_ptr<op::LoopEnd>& loop, bool force_ptr_increment = false) {
        if (loop->get_work_amount() < 2 * loop->get_increment()) {
            loop->set_evaluate_once(true);
            if (force_ptr_increment || loop->has_outer_loop) {
                const auto increment = loop->get_increment();
                std::vector<int64_t> new_finalization_offsets(loop->get_finalization_offsets());
                const auto& apply_increments = loop->get_apply_increment();
                for (auto i = 0; i < new_finalization_offsets.size(); i++) {
                    new_finalization_offsets[i] += increment * apply_increments[i];
                }
                loop->set_finalization_offsets(new_finalization_offsets);
            }
            return true;
        } else {
            return false;
        }
    };
    const auto& ops = m->get_ordered_ops();
    for (auto op = ops.begin(); op < ops.end(); op++) {
        const auto& loop_begin = ov::as_type_ptr<ngraph::snippets::op::LoopBegin>(*op);
        // ignore outer loops and possible manual scalar loops
        if (loop_begin && loop_begin->get_increment() != 1) {
            NodeVector vector_loop, scalar_loop;
            std::shared_ptr<op::LoopEnd> vector_loop_end, scalar_loop_end;
            vector_loop_end = loop_begin->get_loop_end();
            scalar_loop_end = nullptr;
            while (*op != vector_loop_end)
                vector_loop.push_back(*op++);
            vector_loop.push_back(*op);
            const auto work_amount = vector_loop_end->get_work_amount();
            const auto increment = vector_loop_end->get_increment();
            const auto need_scalar_loop = work_amount % increment != 0;
            const auto need_vector_loop = work_amount >= increment;
            // Note, that finalization_offsets could be modified inside optimize_single_evaluation,
            // so need to save them here to cover (evaluate_once vector with non-zero finalization_offsets + scalar)
            std::vector<int64_t> scalar_finalization_offsets = need_scalar_loop ? vector_loop_end->get_finalization_offsets() :
                                                        std::vector<int64_t> {};
            bool vector_evaluate_once = false;
            bool scalar_evaluate_once = false;
            // vector loops are required => Just copy the body, original loop is already a vector one
            if (need_vector_loop) {
                // Note that finalization offsets should be applied after the last iteration.
                // So if there is a scalar loop, then we should apply offsets after it, but not now.
                if (need_scalar_loop)
                    vector_loop_end->set_finalization_offsets(std::vector<int64_t>(scalar_finalization_offsets.size(), 0));
                // force ptr increments if there is at least one scalar loop
                vector_evaluate_once = optimize_single_evaluation(vector_loop_end, need_scalar_loop);
            }
            OV_ITT_TASK_NEXT(GENERATE, "::ScalarLoop")
            // scalar loops are required => transform the body into a scalar representation
            if (need_scalar_loop) {
                NodeMap vector_to_scalar_node_map;
                scalar_loop = ngraph::clone_nodes(vector_loop,  vector_to_scalar_node_map);
                std::transform(scalar_loop.begin(), scalar_loop.end(), scalar_loop.begin(),
                               [](const std::shared_ptr<Node>& n){
                                   if (const auto load = ov::as_type_ptr<ngraph::snippets::op::Load>(n))
                                       load->set_count(1);
                                   else if (const auto store = ov::as_type_ptr<ngraph::snippets::op::Store>(n))
                                       store->set_count(1);
                                   return n;
                               });
                scalar_loop_end = ov::as_type_ptr<op::LoopEnd>(*scalar_loop.rbegin());
                scalar_loop_end->set_finalization_offsets(scalar_finalization_offsets);
                const auto scalar_work_amount = work_amount % increment;
                scalar_loop_end->set_increment(1);
                scalar_loop_end->set_work_amount(scalar_work_amount);
                scalar_loop_end->has_outer_loop = vector_loop_end->has_outer_loop;
                // ptr increment is applied automatically if there is non-empty outer loop
                scalar_evaluate_once = optimize_single_evaluation(scalar_loop_end);
            }
            // Cross-loop optimizations require that both loops are fully initialized,
            // so check need_*_loop again, update optimization flags and lower
            if (need_vector_loop) {
                // scalar loop can't reuse reg_work_amount if vector_evaluate_once==true (since it's not set in this case)
                // likewise, it makes no sense to reuse work_amount_reg if scalar loop is evaluated only once
                vector_loop_end->reuse_work_amount_reg = !vector_evaluate_once && need_scalar_loop && !scalar_evaluate_once;
                lower_ops(vector_loop);
            }
            if (need_scalar_loop) {
                scalar_loop_end->get_loop_begin()->reuse_work_amount_reg = need_vector_loop && vector_loop_end->reuse_work_amount_reg;
                lower_ops(scalar_loop);
            }
        } else {
            lower_ops({*op});
        }
    }

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    // emission
    auto loops2DKernel = std::make_shared<op::Kernel>(std::vector<AllocatedEmitter>{lowered});
    loops2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(op::Kernel::get_type_info_static())(loops2DKernel);

    kernel->emit_code({in, out}, {});

    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    for (auto& op : lowered) {
        op.first->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")
    return target->get_snippet();
}

std::shared_ptr<const TargetMachine> Generator::get_target_machine() const {
    return target;
}

}// namespace snippets
}// namespace ngraph