// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/mark_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

MarkLoops::MarkLoops(size_t vector_size) : Transformation(), m_vector_size(vector_size) {}

bool MarkLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::MarkLoops")
    if (linear_ir.empty())
        return false;

    const auto& lowering_config = linear_ir.get_config();
    const auto& loop_manager = linear_ir.get_loop_manager();
    auto loop_depth = lowering_config.m_loop_depth;

    // Parameters Results or Constants are ignored. They can't be used as a loop starting point
    auto is_not_start_point = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<opset1::Result>(node) ||
               ov::is_type<opset1::Constant>(node) ||
               ov::is_type<opset1::Parameter>(node);
    };

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (is_not_start_point(node))
            continue;

        auto loop_begin_pos = expr_it;
        auto loop_end_pos = loop_begin_pos;

        bool collapse = true;
        do {
            const auto& prev_expr = *loop_end_pos;
            loop_end_pos++;
            // If iterator is the last, we should finish Loop
            if (loop_end_pos == linear_ir.end())
                break;

            // If iterator is the last, we should finish Loop
            const auto& current_expr = *loop_end_pos;
            const auto& current_node = current_expr->get_node();
            if (ov::is_type<opset1::Result>(current_node) ||
                ov::is_type<opset1::Constant>(current_node))
                break;

            // We finish Loop if
            //  - the next expr isn't real customer
            //  - the is conflict between the corresponding ports
            bool is_connected = false;
            bool is_conflicted = false;
            for (size_t i = 0; i < prev_expr->get_output_count(); ++i) {
                const auto& loop_td = prev_expr->output(i);
                const auto consumers = loop_td->get_consumers();
                const auto found = std::find_if(consumers.begin(), consumers.end(), [&loop_end_pos](const ExpressionPort& consumer) {
                    return consumer.get_expr_ptr() == *loop_end_pos;
                });
                if (found != consumers.end()) {
                    if (loop_td->is_conflicted_consumer(*found)) {
                        is_conflicted = true;
                        break;
                    }
                   is_connected = true;
                }
            }
            collapse = is_connected && !is_conflicted;
        } while (collapse);

        loop_manager->mark_loop(loop_begin_pos, loop_end_pos, loop_depth, m_vector_size);
        expr_it = std::prev(loop_end_pos);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
