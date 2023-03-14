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

LoopFusion::LoopFusion() : LinearIRTransformation() {}

bool LoopFusion::can_be_fused(const LoweredLoopManager::LoweredLoopInfoPtr& loop_current, const LoweredLoopManager::LoweredLoopInfoPtr& loop_target) {
    auto current_work_amount = loop_current->m_work_amount;
    auto current_increment = loop_current->m_increment;
    auto target_work_amount = loop_target->m_work_amount;
    auto target_increment = loop_target->m_increment;
    const auto supported_work_amount = current_work_amount == target_work_amount || current_work_amount > 1 || target_work_amount > 1;
    const auto supported_increment = current_increment == target_increment;
    return supported_work_amount && supported_increment;
}

bool LoopFusion::fuse_up(LoweredExprIR& linear_ir,
                         const LoweredExprPort& current_entry_point, const LoweredExprPort& target_exit_point,
                         size_t loop_id, size_t dim_idx,
                         const LoweredLoopManager::LoweredLoopInfoPtr& loop_current,
                         const LoweredLoopManager::LoweredLoopInfoPtr& loop_target,
                         LoweredExprIR::constExprIt& current_loop_begin_pos, LoweredExprIR::constExprIt& current_loop_end_pos) {
    if (!can_be_fused(loop_current, loop_target))
        return false;

    LoweredExprIR::constExprIt target_loop_begin_pos, target_loop_end_pos;
    LoweredLoopManager::get_loop_bounds(linear_ir, loop_target->m_entry_exprs, loop_target->m_exit_exprs, target_loop_begin_pos, target_loop_end_pos);

    // We can fuse Loop_up to Loop_down only in cases when other consumers of Loop_up are after Loop_down
    // Because Loop_up should be explicitly moved before Loop_down in linear IR, and we must save control dependency
    bool is_fusion_allowed = true;
    for (size_t i = 0; i < loop_target->m_exit_exprs.size() && is_fusion_allowed; ++i) {
        const auto target_exit_point = loop_target->m_exit_exprs[i];
        const auto target_exit_expr = target_exit_point.first;
        const auto port = target_exit_point.second;
        const auto output_td = target_exit_expr->get_outputs()[port];
        const auto consumers = linear_ir.get_exprs_by_input(output_td);
        for (const auto& consumer : consumers) {
            if (ov::is_type<opset1::Result>(consumer->get_node()) || consumer == current_entry_point.first)
                continue;
            // The fusing is only valid if target Loop consumer (the Consumer is outside of target Loop)
            // is after current Loop (after Loop_down).
            is_fusion_allowed = std::find(target_loop_begin_pos, target_loop_end_pos, consumer) != target_loop_end_pos || // is inside target Loop
                                std::find(current_loop_end_pos, linear_ir.cend(), consumer) != linear_ir.end();  // is after current Loop
        }
    }

    if (!is_fusion_allowed)
        return false;

    const auto insertion_place = current_loop_begin_pos;
    const auto is_move_needed = target_loop_end_pos != current_loop_begin_pos;
    for (auto it = target_loop_begin_pos; it != target_loop_end_pos;) {
        auto expr_it = it;
        const auto& expr = *expr_it;
        it = std::next(it);
        expr->set_loop_id(loop_id, dim_idx);
        if (is_move_needed)
            linear_ir.splice(insertion_place, expr_it);

        // If after Fusion Result leaves before new Loop, we should insert it after new Loop
        const auto outputs = expr->get_outputs();
        for (const auto& out : outputs) {
            const auto consumers = linear_ir.get_exprs_by_input(out);
            for (const auto& consumer : consumers) {
                if (ov::is_type<opset1::Result>(consumer->get_node())) {
                    const auto result_it = std::find(target_loop_end_pos, current_loop_begin_pos, consumer);
                    if (result_it != current_loop_begin_pos) {
                        linear_ir.splice(std::next(current_loop_end_pos), result_it);
                    }
                }
            }
        }
    }

    // Update current Loop bounds:
    current_loop_begin_pos = target_loop_begin_pos;

    // Update work_amount for Loop (increment is constant because increments must be the identical for fusion):
    loop_current->m_work_amount = std::max(loop_current->m_work_amount, loop_target->m_work_amount);

    // Update entry and exit points in current Loop information
    auto current_entry_points = std::vector<LoweredExprPort>{};
    auto current_exit_points = loop_current->m_exit_exprs;
    auto target_entry_points = loop_target->m_entry_exprs;
    auto target_exit_points = std::vector<LoweredExprPort>{};
    std::copy_if(loop_current->m_entry_exprs.begin(), loop_current->m_entry_exprs.end(), std::back_inserter(current_entry_points),
                 [&current_entry_point](const LoweredExprPort& port) { return port != current_entry_point; } );
    for (const auto& exit_point : loop_target->m_exit_exprs) {
        const auto expr = exit_point.first;
        const auto port = exit_point.second;
        const auto consumers = linear_ir.get_exprs_by_input(expr->get_outputs()[port]);
        bool is_stayed_exit = false;
        for (const auto& consumer_expr : consumers) {
            if (consumer_expr == current_entry_point.first)
                continue;
            is_stayed_exit |= std::find(current_loop_begin_pos, current_loop_end_pos, consumer_expr) != current_loop_end_pos; // is inside Loop
        }

        if (is_stayed_exit)
            target_exit_points.push_back(exit_point);
    }

    std::vector<LoweredExprPort>& new_entries = target_entry_points;
    new_entries.insert(new_entries.end(), current_entry_points.begin(), current_entry_points.end());
    std::vector<LoweredExprPort>& new_exits = target_exit_points;
    new_exits.insert(new_exits.end(), current_exit_points.begin(), current_exit_points.end());

    loop_current->m_entry_exprs = new_entries;
    loop_current->m_exit_exprs = new_exits;

    return true;
}

bool LoopFusion::fuse_down(LoweredExprIR& linear_ir,
                           const LoweredExprPort& current_exit_point, const LoweredExprPort& target_entry_point,
                           size_t loop_id, size_t dim_idx,
                           const LoweredLoopManager::LoweredLoopInfoPtr& loop_current,
                           const LoweredLoopManager::LoweredLoopInfoPtr& loop_target,
                           LoweredExprIR::constExprIt& current_loop_begin_pos, LoweredExprIR::constExprIt& current_loop_end_pos) {
    if (!can_be_fused(loop_current, loop_target))
        return false;

    // We can fuse Loop_down to Loop_up only in cases when other parents of Loop_down are before Loop_up
    // Because Loop_down should be explicitly moved after Loop_up in linear IR, and we must save control dependency
    bool is_fusion_allowed = true;
    for (size_t i = 0; i < loop_target->m_entry_exprs.size() && is_fusion_allowed; ++i) {
        const auto target_entry_point = loop_target->m_entry_exprs[i];
        const auto target_entry_expr = target_entry_point.first;
        const auto port = target_entry_point.second;
        const auto input_td = target_entry_expr->get_inputs()[port];
        const auto parent_expr = linear_ir.get_expr_by_output(input_td);
        if (ov::is_type<opset1::Parameter>(parent_expr->get_node()) || parent_expr == current_exit_point.first)
            continue;
        is_fusion_allowed = std::find(linear_ir.cbegin(), current_loop_begin_pos, parent_expr) != current_loop_begin_pos;
    }

    if (!is_fusion_allowed)
        return false;

    LoweredExprIR::constExprIt target_loop_begin_pos, target_loop_end_pos;
    LoweredLoopManager::get_loop_bounds(linear_ir, loop_target->m_entry_exprs, loop_target->m_exit_exprs, target_loop_begin_pos, target_loop_end_pos);
    const auto insertion_place = current_loop_end_pos;
    const auto is_move_needed = insertion_place != target_loop_begin_pos;
    for (auto it = target_loop_begin_pos; it != target_loop_end_pos;) {
        auto expr_it = it;
        const auto& expr = *expr_it;
        it = std::next(it);
        expr->set_loop_id(loop_id, dim_idx);
        if (is_move_needed)
            linear_ir.splice(insertion_place, expr_it);
    }

    // Update current Loop bounds:
    if (!is_move_needed)
        current_loop_end_pos = target_loop_end_pos;

    // Update work_amount for Loop (increment is constant because increments must be the identical for fusion):
    loop_current->m_work_amount = std::max(loop_current->m_work_amount, loop_target->m_work_amount);

    // Update entry and exit points in current Loop information
    auto current_entry_points = loop_current->m_entry_exprs;
    auto current_exit_points = std::vector<LoweredExprPort>{};
    auto target_entry_points = std::vector<LoweredExprPort>{};
    auto target_exit_points = loop_target->m_exit_exprs;
    std::copy_if(loop_target->m_entry_exprs.begin(), loop_target->m_entry_exprs.end(), std::back_inserter(target_entry_points),
                 [&target_entry_point](const LoweredExprPort& port) { return port != target_entry_point; } );
    for (const auto& exit_point : loop_current->m_exit_exprs) {
        const auto expr = exit_point.first;
        const auto port = exit_point.second;
        const auto consumers = linear_ir.get_exprs_by_input(expr->get_outputs()[port]);
        bool is_stayed_exit = false;
        for (const auto& consumer_expr : consumers) {
            if (consumer_expr == target_entry_point.first)
                continue;
            is_stayed_exit |= std::find(current_loop_begin_pos, current_loop_end_pos, consumer_expr) == current_loop_end_pos;
        }

        if (is_stayed_exit)
            current_exit_points.push_back(exit_point);
    }

    std::vector<LoweredExprPort>& new_entries = current_entry_points;
    new_entries.insert(new_entries.end(), target_entry_points.begin(), target_entry_points.end());
    std::vector<LoweredExprPort>& new_exits = current_exit_points;
    new_exits.insert(new_exits.end(), target_exit_points.begin(), target_exit_points.end());

    loop_current->m_entry_exprs = new_entries;
    loop_current->m_exit_exprs = new_exits;

    return true;
}

bool LoopFusion::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::LoopFusion")
    if (linear_ir.empty())
        return false;

    const auto& lowering_config = linear_ir.get_config();
    const auto& loop_manager = linear_ir.get_loop_manager();
    std::vector<size_t> prev_expr_loops;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<opset1::Parameter>(node) ||
            ov::is_type<opset1::Constant>(node) ||
            ov::is_type<opset1::Result>(node) ||
            ov::is_type<op::Brgemm>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto expr_loops = expr->get_loop_ids();
        const auto loop_depth = expr_loops.size();
        size_t diff_idx = 0;
        if (prev_expr_loops.empty()) {
            prev_expr_loops = expr_loops;
        } else {
            OPENVINO_ASSERT(loop_depth == prev_expr_loops.size(),
                            "Expressions in Linear IR must have the same count of Loop identifiers");
            for (; diff_idx < loop_depth; ++diff_idx) {
                if (expr_loops[diff_idx] != prev_expr_loops[diff_idx])
                    break;
            }
        }

        for (size_t dim_idx = diff_idx; dim_idx < loop_depth; ++dim_idx) {
            const auto loop_id = expr_loops[dim_idx];
            if (loop_id == LoweredLoopManager::EMPTY_ID)
                continue;

            const auto loop_info = loop_manager->get(loop_id);
            LoweredExprIR::constExprIt loop_begin_pos, loop_end_pos;
            LoweredLoopManager::get_loop_bounds(linear_ir, loop_info->m_entry_exprs, loop_info->m_exit_exprs, loop_begin_pos, loop_end_pos);

            // We fuse Loops on inputs till we can do it.
            // After that we fuse Loops on outputs will we can do it.
            // If we have fused on outputs we should verify possible fusions on inputs again because of new entry points
            while (true) {
                auto entry_points = loop_info->m_entry_exprs;
                bool was_fusion_up = false;
                for (size_t in_port = 0; in_port < entry_points.size() && !was_fusion_up; ++in_port) {
                    const auto entry_point = entry_points[in_port];
                    const auto entry_expr = entry_point.first;
                    const auto port = entry_point.second;
                    const auto input_td = entry_expr->get_inputs()[port];
                    const auto parent_expr = linear_ir.get_expr_by_output(input_td);
                    const auto parent = parent_expr->get_node();
                    if (ov::is_type<opset1::Constant>(parent) ||
                        ov::is_type<opset1::Parameter>(parent) ||
                        ov::is_type<op::Buffer>(parent) ||
                        ov::is_type<op::Brgemm>(parent)) {
                        continue;
                    }
                    const auto loop_ids_target = parent_expr->get_loop_ids();
                    OPENVINO_ASSERT(loop_depth == loop_ids_target.size(),
                                    "Expressions in Linear IR must have the same count of Loop identifiers");
                    const auto loop_id_target = loop_ids_target[dim_idx];
                    OPENVINO_ASSERT(loop_id != loop_id_target,
                                    "Loops cannot have parents of entry points with the same identifier");
                    const auto loop_info_target = loop_manager->get(loop_id_target);

                    const auto target_exit_port = LoweredExprPort{parent_expr, parent_expr->get_output_port(input_td)};
                    if (fuse_up(linear_ir, entry_point, target_exit_port, loop_id, dim_idx, loop_info, loop_info_target, loop_begin_pos, loop_end_pos)) {
                        was_fusion_up = true;
                        loop_manager->remove(loop_id_target);
                        linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                                            "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
                        // Need to check for possible fusion again because of new input expressions for Loop
                        break;
                    }
                }

                // If Loops were fused, we should check for possible fusion again
                if (was_fusion_up)
                    continue;

                auto exit_points = loop_info->m_exit_exprs;
                bool was_fusion_down = false;
                for (size_t out_port = 0; out_port < exit_points.size() && !was_fusion_down; ++out_port) {
                    const auto exit_point = exit_points[out_port];
                    const auto exit_expr = exit_point.first;
                    const auto port = exit_point.second;
                    const auto output_td = exit_expr->get_outputs()[port];
                    const auto consumer_exprs = linear_ir.get_exprs_by_input(output_td);
                    for (const auto& consumer_expr : consumer_exprs) {
                        const auto consumer = consumer_expr->get_node();
                        if (ov::is_type<opset1::Result>(consumer) ||
                            ov::is_type<op::Buffer>(consumer) ||
                            ov::is_type<op::Brgemm>(consumer)) {
                            continue;
                        }

                        const auto loop_ids_target = consumer_expr->get_loop_ids();
                        OPENVINO_ASSERT(loop_depth == loop_ids_target.size(),
                                        "Expressions in Linear IR must have the same count of Loop identifiers");
                        // The exit point of Loop can have several consumers where some of them can be in this Loop as well
                        // So we skip this consumer.
                        const auto loop_id_target = loop_ids_target[dim_idx];
                        if (loop_id == loop_id_target)
                            continue;

                        const auto loop_info_target = loop_manager->get(loop_id_target);
                        const auto target_entry_port = LoweredExprPort{consumer_expr, consumer_expr->get_input_port(output_td)};
                        if (fuse_down(linear_ir, exit_point, target_entry_port, loop_id, dim_idx, loop_info, loop_info_target, loop_begin_pos, loop_end_pos)) {
                            was_fusion_down = true;
                            loop_manager->remove(loop_id_target);
                            linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                                                "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
                            // Need to check for possible fusion again because of new input expressions for Loop
                            break;
                        }
                    }
                }

                // We iterated by each exit point and didn't fuse new Loops -> we can finish check for possible fusions on outputs.
                if (!was_fusion_down)
                    break;
            }
        }
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

