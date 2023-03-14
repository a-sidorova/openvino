// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_buffer.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"


namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

InsertBuffer::InsertBuffer(size_t buffer_allocation_rank)
    : LinearIRTransformation(), m_buffer_allocation_rank(buffer_allocation_rank) {}

void InsertBuffer::insertion(LoweredExprIR& linear_ir, size_t loop_id,
                             const std::vector<LoweredExprPort>& loop_entries, const std::vector<LoweredExprPort>& loop_exits) {
    LoweredExprIR::constExprIt loop_begin_pos, loop_end_pos;
    LoweredLoopManager::get_loop_bounds(linear_ir, loop_entries, loop_exits, loop_begin_pos, loop_end_pos);

    for (const auto& entry_point : loop_entries) {
        const auto expr = entry_point.first;
        const auto port = entry_point.second;
        const auto node = expr->get_node();
        const auto input_td = expr->get_inputs()[port];
        const auto parent_expr = linear_ir.get_expr_by_output(input_td);
        const auto parent = parent_expr->get_node();
        if (ov::is_type<op::Buffer>(parent) ||
            ov::is_type<op::VectorBuffer>(parent) ||
            ov::is_type<opset1::Parameter>(parent) ||
            ov::is_type<opset1::Constant>(parent))
            continue;

        bool is_buffer_needed = ov::is_type<op::Brgemm>(parent) || ov::is_type<op::Brgemm>(node);
        if (!is_buffer_needed) {
            const auto current_loops = expr->get_loop_ids();
            const auto parent_loops = parent_expr->get_loop_ids();
            const auto current_loop_count = current_loops.size();
            const auto parent_loop_count = parent_loops.size();
            OPENVINO_ASSERT(current_loop_count == parent_loop_count);
            const auto current_loop_lvl = std::distance(current_loops.begin(), std::find(current_loops.begin(), current_loops.end(), loop_id));
            for (size_t i = current_loop_lvl; i < current_loop_count; i++) {
                if (current_loops[i] != parent_loops[i] &&
                    current_loops[i] != LoweredLoopManager::EMPTY_ID &&
                    parent_loops[i] != LoweredLoopManager::EMPTY_ID) {
                    is_buffer_needed = true;
                    break;
                }
            }
        }

        if (is_buffer_needed) {
            const auto parent_port = parent_expr->get_output_port(input_td);
            auto buffer = std::make_shared<op::Buffer>(parent->output(parent_port), m_buffer_allocation_rank);

            const auto td = std::make_shared<TensorDescriptor>(input_td->get_tensor(),
                                                               std::vector<size_t>{},  // or copy?
                                                               input_td->get_layout());
            const std::vector<TensorDescriptorPtr> buffer_outs = { td };
            const std::vector<TensorDescriptorPtr> parent_outs = { input_td };
            linear_ir.insert(loop_begin_pos, std::make_shared<LoweredExpr>(buffer, parent_outs, buffer_outs));
            linear_ir.replace_input(expr, input_td, td);
        }
    }

    for (const auto& exit_point : loop_exits) {
        const auto expr = exit_point.first;
        const auto port = exit_point.second;
        const auto node = expr->get_node();
        const auto output_td = expr->get_outputs()[port];
        const auto child_exprs = linear_ir.get_exprs_by_input(output_td);
        const auto current_loops = expr->get_loop_ids();
        const auto current_loop_count = current_loops.size();
        const std::vector<TensorDescriptorPtr> node_outs = {output_td};

        std::set<LoweredExprPtr> potential_consumers;
        std::set<LoweredExprPtr> buffers;
        for (const auto &child_expr : child_exprs) {
            const auto child = child_expr->get_node();
            if (ov::is_type<opset1::Result>(child))
                continue;
            if (ov::is_type<op::Buffer>(child)) {
                buffers.insert(child_expr);
                continue;
            }
            if (ov::is_type<op::Brgemm>(child) || ov::is_type<op::Brgemm>(node)) {
                potential_consumers.insert(child_expr);
                continue;
            }

            const auto child_loops = child_expr->get_loop_ids();
            const auto child_loop_count = child_loops.size();
            OPENVINO_ASSERT(current_loop_count == child_loop_count);
            const auto current_loop_lvl = std::distance(current_loops.begin(), std::find(current_loops.begin(), current_loops.end(), loop_id));
            for (size_t i = 0; i <= current_loop_lvl; i++) {
                if (current_loops[i] != child_loops[i] &&
                    current_loops[i] != LoweredLoopManager::EMPTY_ID &&
                    child_loops[i] != LoweredLoopManager::EMPTY_ID) {
                    potential_consumers.insert(child_expr);
                    break;
                }
            }
        }

        if (!potential_consumers.empty()) {
            // If some of children from one common port are different Buffers,
            // we should remove them to insert one common Buffer on one common port
            if (!buffers.empty()) {
                for (const auto& buffer : buffers) {
                    const auto buffer_out = buffer->get_outputs().front();
                    const auto buffer_consumers = linear_ir.get_exprs_by_input(buffer_out);
                    for (const auto& consumer : buffer_consumers)
                        linear_ir.replace_input(consumer, buffer_out, output_td);
                    potential_consumers.insert(buffer_consumers.begin(), buffer_consumers.end());
                    linear_ir.erase(std::find(linear_ir.begin(), linear_ir.end(), buffer));
                }
            }

            auto buffer = std::make_shared<op::Buffer>(node->output(port), m_buffer_allocation_rank);
            const auto td = std::make_shared<TensorDescriptor>(output_td->get_tensor(),
                                                               std::vector<size_t>{},
                                                               output_td->get_layout());
            // We cannot insert Node output tensor on Buffer output because not all consumers of Node needs Buffer
            //  Example:
            //       Add
            //      /   \  <- It should be the same TD
            //  Result   Buffer
            //             |    <- It should be new TD
            //            Relu
            const std::vector<TensorDescriptorPtr> buffer_outs = {td};
            linear_ir.insert(loop_end_pos,
                             std::make_shared<LoweredExpr>(buffer, node_outs, buffer_outs));
            for (const auto consumer : potential_consumers) {
                linear_ir.replace_input(consumer, output_td, td);
            }
        }
    }
}

bool InsertBuffer::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InsertBuffer")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    for (const auto loop_id : loop_manager->get_identifies()) {
        const auto loop_info = loop_manager->get(loop_id);
        const auto loop_entries = loop_info->m_entry_exprs;
        const auto loop_exits = loop_info->m_exit_exprs;
        insertion(linear_ir, loop_id, loop_entries, loop_exits);

        linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                            "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    }

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto node = (*expr_it)->get_node();
        if (!ov::is_type<op::Brgemm>(node))
            continue;

        std::vector<LoweredExprPort> loop_entries = {{expr, 0}, {expr, 1}};
        std::vector<LoweredExprPort> loop_exits = {{expr, 0}};

        insertion(linear_ir, LoweredLoopManager::EMPTY_ID, loop_entries, loop_exits);
        linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                            "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

