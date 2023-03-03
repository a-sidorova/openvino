// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

InsertLoops::InsertLoops(size_t vector_size) : LinearIRTransformation(), m_vector_size(vector_size) {}

LoweredExprIR::constExprIt InsertLoops::insert_one_loop(LoweredExprIR& linear_ir,
                                                        LoweredExprIR::constExprIt loop_begin_pos,
                                                        LoweredExprIR::constExprIt loop_end_pos,
                                                        size_t work_amount,
                                                        size_t work_amount_increment) {
    const auto& loop_begin = std::make_shared<op::LoopBegin>();
    const auto& loop_begin_expr = std::make_shared<LoweredExpr>(loop_begin, std::vector<TensorDescriptorPtr>{});
    loop_begin_pos = linear_ir.insert(loop_begin_pos, loop_begin_expr);

    const auto& loop_end = std::make_shared<op::LoopEnd>(
            OutputVector{loop_begin}, work_amount, work_amount_increment, std::vector<int64_t>{}, std::vector<int64_t>{});
    const auto& loop_end_expr = std::make_shared<LoweredExpr>(loop_end);
    linear_ir.insert(loop_end_pos, loop_end_expr);
    return loop_begin_pos;
}

void InsertLoops::insertion(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos,
                            size_t loop_depth, size_t vector_size) {
    // Note: currently we simply take out td of the last expr in the loop. If needed,
    // this can be generalized for loops with multiple different out td's.
    const auto& out_td = std::prev(loop_end_pos)->get()->get_outputs().front();
    const auto& tensor_out = out_td->get_tensor();
    const auto& subtensor_in = loop_begin_pos->get()->get_outputs().front()->get_subtensor();

    for (size_t idx = 0; idx < loop_depth; ++idx) {
        OPENVINO_ASSERT(idx < tensor_out.size(), "Incorrect indexes of Loop for insertion");
        const auto work_amount = *(tensor_out.rbegin() + idx);
        const auto increment = idx < subtensor_in.size() ? *(subtensor_in.rbegin() + idx)
                                                       : idx == 0 ? vector_size : 1lu;
        loop_begin_pos = insert_one_loop(linear_ir, loop_begin_pos, loop_end_pos, work_amount, increment);
    }
}

bool InsertLoops::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InsertLoops")
    if (linear_ir.empty())
        return false;

    const auto& lowering_config = linear_ir.get_config();
    auto loop_depth = lowering_config.m_loop_depth;

    // Parameters Results or Constants are ignored. They can't be used as a loop starting point
    auto is_not_start_point = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<opset1::Constant>(node) ||
               ov::is_type<opset1::Result>(node) ||
               ov::is_type<opset1::Parameter>(node) ||
               ov::is_type<opset1::Softmax>(node) ||
               ov::is_type<op::Brgemm>(node);
    };

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& node = expr_it->get()->get_node();
        if (is_not_start_point(node))
            continue;

        // Skip existing loops? Softmax history

        auto loop_begin_pos = expr_it;
        auto loop_end_pos = loop_begin_pos;

        const auto& outputs = expr_it->get()->get_outputs();
        const auto& loop_inner_layout = outputs.front()->get_layout();
        const auto& loop_inner_subtensor = outputs.front()->get_subtensor();

        bool is_inside = true;
        do {
            const auto& prev_node = loop_end_pos->get()->get_node();
            loop_end_pos++;
            // If iterator is the last, we should finish Loop
            if (loop_end_pos == linear_ir.end())
                break;

            // If iterator is the last, we should finish Loop
            const auto& current_node = loop_end_pos->get()->get_node();
            if (ov::is_type<op::Brgemm>(current_node) ||
                ov::is_type<opset1::Softmax>(current_node) ||
                ov::is_type<opset1::Result>(current_node))
                break;

            // If the next expr isn't real customer of prev expr we should finish Loop
            std::set<ov::Input<ov::Node>> prev_node_customers;
            for (const auto& output : prev_node->outputs()) {
                const auto target_inputs = output.get_target_inputs();
                prev_node_customers.insert(target_inputs.begin(), target_inputs.end());
            }
            auto compare = [&current_node](const ov::Input<ov::Node>& input){ return input.get_node()->shared_from_this() == current_node;};
            if (std::none_of(prev_node_customers.begin(), prev_node_customers.end(), compare))
                break;

            if (ov::is_type<opset1::Constant>(current_node))
                continue;

            const auto& ins = loop_end_pos->get()->get_inputs();

            const auto& layout = ins.front()->get_layout();
            const auto& subtensor = ins.front()->get_subtensor();
            is_inside &= layout == loop_inner_layout && subtensor == loop_inner_subtensor;
        } while (is_inside);

        insertion(linear_ir, loop_begin_pos, loop_end_pos, loop_depth, m_vector_size);
        expr_it = std::prev(loop_end_pos);
        linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                            "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    }
    linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                        "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

