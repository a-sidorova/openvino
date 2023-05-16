// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/move_result_out_of_loop.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

bool MoveResultOutOfLoop::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::MoveResultOutOfLoop")
    if (linear_ir.empty())
        return false;

    bool modified = false;
    const auto loop_manager = linear_ir.get_loop_manager();
    // Visit expressions in reverse order, so we'll move Result to an already visited area.
    // This is needed to avoid extra hits, when we match to the same Result twice
    for (auto expr_it = linear_ir.crbegin(); expr_it != linear_ir.crend(); expr_it++) {
        const auto& forward_it = std::prev(expr_it.base());
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (!ov::is_type<opset1::Result>(node)) {
            continue;
        }

        const auto input_td = expr->get_inputs().front();
        const auto parent_expr = linear_ir.get_expr_by_output(input_td).expr;
        const auto parent_loop_ids = parent_expr->get_loop_ids();
        int outer_loop_id = static_cast<int>(parent_loop_ids.size()) - 1;
        for (; outer_loop_id >= 0; --outer_loop_id) {
            if (parent_loop_ids[outer_loop_id] != LoweredExpr::LOOP_NULL_ID) {
                break;
            }
        }

        // Parent is out of Loop: just verify that Result is after Parent
        if (outer_loop_id < 0) {
            const auto parent_it = std::find(forward_it, linear_ir.cend(), parent_expr);
            // If Parent is found after Result, we should move Result
            if (parent_it != linear_ir.cend()) {
                const auto insertion_pos = std::next(parent_it);
                const auto result_it = forward_it;
                expr_it = std::prev(expr_it);  // save iterator before moving
                linear_ir.move(result_it, insertion_pos);
                modified = true;
            }
            continue;
        }

        LoweredExprIR::constExprIt loop_begin_pos, loop_end_pos;
        loop_manager->get_loop_bounds(linear_ir, parent_loop_ids[outer_loop_id], loop_begin_pos, loop_end_pos);
        // If the Result isn't found after Outer LoopEnd, need to move it to there
        if (std::find(loop_end_pos, linear_ir.cend(), expr) == linear_ir.cend()) {
            expr_it = std::prev(expr_it);  // save iterator before moving
            linear_ir.move(forward_it, loop_end_pos);
            modified = true;
        }
    }

    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
