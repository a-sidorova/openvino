// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_load_store.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

InsertLoadStore::InsertLoadStore(size_t vector_size) : m_vector_size(vector_size) {}

void InsertLoadStore::update_loops(const LoweredLoopManagerPtr& loop_manager, const std::vector<size_t>& loop_ids,
                                   const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry) {
    for (auto loop_id : loop_ids) {
        if (loop_id != LoweredLoopManager::EMPTY_ID)
            update_loop(loop_manager->get(loop_id), actual_port, target_ports, is_entry);
    }
}

void InsertLoadStore::update_loop(const LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                                  const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry) {
    auto& ports = is_entry ? loop_info->m_entry_exprs : loop_info->m_exit_exprs;
    auto port_it = std::find(ports.begin(), ports.end(), actual_port);
    if (port_it == ports.end())
        return;
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

void InsertLoadStore::insert_load(LoweredExprIR& linear_ir,
                                  const LoweredLoopManagerPtr& loop_manager,
                                  const LoweredExprPort& entry_point,
                                  LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos) {
    const auto expr = entry_point.first;
    const auto port = entry_point.second;
    const auto node = expr->get_node();
    if (ov::is_type<op::Load>(node)) {
        return;
    }

    const auto input_td = expr->get_inputs()[port];
    const auto parent_expr = linear_ir.get_expr_by_output(input_td);
    const auto parent = parent_expr->get_node();
    const auto parent_port = parent_expr->get_output_port(input_td);

    // TODO: Need to align: Can we have IO of Loops without GPR (for example, ReduceMax/ReduceSum)
    if (!ov::is_type<op::Buffer>(parent) && !ov::is_type<opset1::Parameter>(parent)) {
        return;
    }

    const auto load_td = std::make_shared<TensorDescriptor>(input_td->get_tensor(),
                                                            input_td->get_subtensor(),
                                                            input_td->get_layout());
    const auto load = std::make_shared<op::Load>(parent->output(parent_port), m_vector_size);
    const auto load_outs = std::vector<TensorDescriptorPtr>{ load_td };
    const auto param_outs = std::vector<TensorDescriptorPtr>{ input_td };
    const auto load_expr = std::make_shared<LoweredExpr>(load, param_outs, load_outs);
    linear_ir.insert(std::find(loop_begin_pos, loop_end_pos, expr), load_expr);
    linear_ir.replace_input(expr, input_td, load_td);
    const auto new_entry_point = LoweredExprPort{load_expr, 0};
    // Copy Loop IDs
    const auto loop_ids = expr->get_loop_ids();
    load_expr->set_loop_ids(loop_ids);
    update_loops(loop_manager, loop_ids, entry_point, {new_entry_point}, true);
}

void InsertLoadStore::insert_store(LoweredExprIR& linear_ir,
                                   const LoweredLoopManagerPtr& loop_manager,
                                   const LoweredExprPort& exit_point,
                                   LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos) {
    const auto expr = exit_point.first;
    const auto port = exit_point.second;
    const auto node = expr->get_node();
    if (ov::is_type<op::Store>(node)) {
        return;
    }

    std::vector<LoweredExprPort> new_exit_exprs;
    const auto output_td = expr->get_outputs()[port];
    const auto child_exprs = linear_ir.get_exprs_by_input(output_td);
    const auto loop_ids = expr->get_loop_ids();
    auto store_pos = std::next(std::find(loop_begin_pos, linear_ir.cend(), expr));
    for (const auto& child_expr : child_exprs) {
        const auto child = child_expr->get_node();
        const auto port = child_expr->get_input_port(output_td);

        // TODO: Need to align: Can we have IO of Loops without GPR (for example, ReduceMax/ReduceSum)
        if (!ov::is_type<op::Buffer>(child) && !ov::is_type<opset1::Result>(child)) {
            continue;
        }

        const auto store_td = std::make_shared<TensorDescriptor>(output_td->get_tensor(),
                                                                 output_td->get_subtensor(),
                                                                 output_td->get_layout());
        auto store = std::make_shared<op::Store>(node->output(port), m_vector_size);
        const std::vector<TensorDescriptorPtr> parent_outs { output_td };
        const std::vector<TensorDescriptorPtr> store_outs { store_td };
        const auto store_expr = std::make_shared<LoweredExpr>(store, parent_outs, store_outs);
        linear_ir.insert(store_pos, store_expr);
        linear_ir.replace_input(child_expr, output_td, store_td);
        // Copy Loop IDS
        store_expr->set_loop_ids(loop_ids);
        // Update entry expressions.
        new_exit_exprs.push_back({store_expr, 0});
    }

    update_loops(loop_manager, loop_ids, exit_point, new_exit_exprs, false);
}

bool InsertLoadStore::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InsertLoadStore")

    const auto& lowering_config = linear_ir.get_config();
    const auto& loop_manager = linear_ir.get_loop_manager();

    auto loop_begin_pos = linear_ir.cbegin();
    auto loop_end_pos = linear_ir.cend();
    std::vector<size_t> prev_expr_loops;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto &node = expr->get_node();
        if (ov::is_type<opset1::Parameter>(node) ||
            ov::is_type<opset1::Constant>(node) ||
            ov::is_type<opset1::Result>(node) ||
            ov::is_type<op::Brgemm>(node))
            continue;

        // Found Inner Loop
        const auto expr_loops = expr->get_loop_ids();
        if (prev_expr_loops == expr_loops || expr_loops.empty()) {
            continue;
        }
        prev_expr_loops = expr_loops;
        const auto loop_depth = expr_loops.size();
        size_t loop_id = LoweredLoopManager::EMPTY_ID;
        for (int i = loop_depth - 1; i >= 0; --i) {
            if (expr_loops[i] != LoweredLoopManager::EMPTY_ID) {
                loop_id = expr_loops[i];
                break;
            }
        }

        const auto& loop_info = loop_manager->get(loop_id);
        const auto entry_exprs = loop_info->m_entry_exprs;
        const auto exit_exprs = loop_info->m_exit_exprs;
        LoweredLoopManager::get_loop_bounds(linear_ir, entry_exprs, exit_exprs, loop_begin_pos, loop_end_pos);

        for (const auto& entry_point : entry_exprs) {
            insert_load(linear_ir, loop_manager, entry_point, loop_begin_pos, loop_end_pos);
            linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                                "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
        }
        for (const auto& exit_point : exit_exprs) {
            insert_store(linear_ir, loop_manager, exit_point, loop_begin_pos, loop_end_pos);
            linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                                "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
        }
    }

    linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                        "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

