// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/loop_fusion.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

namespace {

auto get_execution_order(const LoweredExprIR& linear_ir) -> std::vector<LoweredExprIR::constExprIt> {
    std::vector<LoweredExprIR::constExprIt> order;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto node = expr_it->get()->get_node();
        if (ov::is_type<ov::op::v0::Parameter>(node) ||
            ov::is_type<ov::op::v0::Constant>(node) ||
            ov::is_type<ov::op::v0::Result>(node) ||
            ov::is_type<op::LoopEnd>(node)) {
            order.push_back(expr_it);
        }
        if (ov::is_type<op::LoopBegin>(node)) {
            continue;
        }

        auto prev_expr = std::prev(expr_it);
        auto next_expr = std::next(expr_it);
        while (!ov::is_type<op::LoopBase>((*prev_expr)->get_node()) ||
               !ov::is_type<opset1::Parameter>((*prev_expr)->get_node()) ||
               !ov::is_type<opset1::Constant>((*prev_expr)->get_node())) {
            prev_expr = std::prev(expr_it);
        }
        while (!ov::is_type<op::LoopBase>((*next_expr)->get_node()) ||
               !ov::is_type<opset1::Result>((*next_expr)->get_node())) {
            next_expr = std::next(next_expr);
        }
        const bool node_in_loop = ov::is_type<op::LoopBegin>((*prev_expr)->get_node()) &&
                                  ov::is_type<op::LoopEnd>((*next_expr)->get_node());
        if (!node_in_loop) {
            order.push_back(expr_it);
        }
    }

    return order;
}

auto fuse(LoweredExprIR& linear_ir,
          LoweredExprIR::constExprIt loop_begin_0, LoweredExprIR::constExprIt loop_end_0,
          LoweredExprIR::constExprIt loop_begin_1, LoweredExprIR::constExprIt loop_end_1,
          size_t work_amount, size_t increment) -> bool {
    const auto insertion_place = std::next(loop_begin_1);
    for (auto it = std::next(loop_begin_0); it != loop_end_0;) {
        auto expr_it = it;
        it = std::next(it);
        linear_ir.splice(insertion_place, expr_it);

        if (ov::is_type<op::LoopBase>((*expr_it)->get_node()))
            continue;

        const auto outputs = (*expr_it)->get_outputs();
        for (const auto& out : outputs) {
            const auto consumers = linear_ir.get_exprs_by_input(out);
            for (const auto& consumer : consumers) {
                if (ov::is_type<opset1::Result>(consumer->get_node())) {
                    // If after Fusion Result leaves before new Loop, we should insert it after new Loop
                    const auto result_it = std::find(linear_ir.cbegin(), loop_begin_1, consumer);
                    if (result_it != loop_begin_1) {
                        linear_ir.splice(std::next(loop_end_1), result_it);
                    }
                }
            }
        }
    }

    linear_ir.erase(loop_begin_0);
    linear_ir.erase(loop_end_0);

    const auto loop_begin = ov::as_type_ptr<op::LoopBegin>((*loop_begin_1)->get_node());
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>((*loop_end_1)->get_node());
    OPENVINO_ASSERT(loop_begin && loop_end, "Fusion Loops expects LoopBegin and LoopEnd after fusion");
    loop_end->set_work_amount(work_amount);
    loop_end->set_increment(increment);

    return true;
}
}  // namespace

LoopFusion::LoopFusion() : LinearIRTransformation() {}

bool LoopFusion::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::LoopFusion")
    if (linear_ir.empty())
        return false;

   // auto execution_order = get_execution_order(linear_ir);

    bool is_modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end();) {
        const auto loop_begin = ov::as_type_ptr<op::LoopBegin>((*expr_it)->get_node());
        if (!loop_begin) {
            expr_it = std::next(expr_it);
            continue;
        }

        const auto loop_end = loop_begin->get_loop_end();
        auto loop_begin_expr_it = expr_it;
        auto loop_end_expr_it = std::find(loop_begin_expr_it, linear_ir.end(), linear_ir.get_expr_by_node(loop_end));
        OPENVINO_ASSERT(loop_end_expr_it != linear_ir.end(), "LoopBegin must have corresponding LoopEnd in Linear IR!");
        auto current_body_ops = LoweredExprIR::get_loop_body(loop_begin_expr_it, loop_end_expr_it);

        // TODO: Decomposed Transpose doesn't support fusing of inner Loops
        if (current_body_ops.size() == 2 && ov::is_type<op::LoadReshape>(current_body_ops[0]) && ov::is_type<op::Store>(current_body_ops[1])) {
            // Skip the Transpose Loop
            expr_it = std::next(loop_end_expr_it);
            continue;
        }

        bool is_fused = false;
        for (auto fusing_expr_it = std::next(loop_end_expr_it); fusing_expr_it != linear_ir.end() && !is_fused; fusing_expr_it++) {
            if (const auto fusing_loop_begin = ov::as_type_ptr<op::LoopBegin>((*fusing_expr_it)->get_node())) {
                const auto fusing_loop_end = fusing_loop_begin->get_loop_end();
                auto fusing_loop_begin_expr_it = fusing_expr_it;
                auto fusing_loop_end_expr_it = std::find(fusing_loop_begin_expr_it, linear_ir.end(), linear_ir.get_expr_by_node(fusing_loop_end));
                OPENVINO_ASSERT(fusing_loop_end_expr_it != linear_ir.end(), "LoopBegin must have corresponding LoopEnd in Linear IR!");

                auto fusing_body_ops = LoweredExprIR::get_loop_body(fusing_loop_begin_expr_it, fusing_loop_end_expr_it);
                bool is_needed = false;
                for (size_t i = 0; i < fusing_body_ops.size() && !is_needed; ++i) {
                    const auto op = fusing_body_ops[i];
                    // todo: rewrite on td
                    for (const auto& input : op->input_values()) {
                        const auto input_node = input.get_node_shared_ptr();
                        is_needed |= std::find(current_body_ops.begin(), current_body_ops.end(), input_node) != current_body_ops.end();
                    }
                }

                // If there aren't dependencies between Loops (no consumers and parents),
                // we should skip this loop with body (with possible loops inside)
                if (!is_needed) {
                    fusing_expr_it = fusing_loop_end_expr_it;
                    continue;
                }

                // TODO: Check for cycle dependency

                auto current_work_amount = loop_end->get_work_amount();
                auto current_increment = loop_end->get_increment();
                auto fusing_work_amount = fusing_loop_end->get_work_amount();
                auto fusing_increment = fusing_loop_end->get_increment();
                if ((current_work_amount != fusing_work_amount && current_work_amount > 1 && fusing_work_amount > 1) ||
                    (current_increment != fusing_increment && current_increment > 1 && fusing_increment > 1)) {
                    fusing_expr_it = fusing_loop_end_expr_it;
                    continue;
                }

                // Before save the state of iterators because this iterator is removed during fusion
                expr_it = std::prev(loop_begin_expr_it);

                is_fused = fuse(linear_ir,
                                loop_begin_expr_it, loop_end_expr_it,
                                fusing_loop_begin_expr_it, fusing_loop_end_expr_it,
                                std::max(current_work_amount, fusing_work_amount),
                                std::max(current_increment, fusing_increment));
                linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                                    "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
            }
        }

        // If the Loop hasn't been fused into another, we should manually increment iterator
        if (!is_fused) {
            expr_it = std::next(expr_it);
        }

        is_modified |= is_fused;
    }

    return is_modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

