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

void InsertBuffer::insertion(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos) {
    auto expr_it = loop_begin_pos;
    do {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopBase>(node) || ov::is_type<op::Buffer>(node)) {
            expr_it++;
            continue;
        }

        // Insert Buffer before Loop if the nodes are connected from different Loops or at least one of them is Brgemm
        for (const auto& input_td : expr->get_inputs()) {
            const auto parent_expr = linear_ir.get_expr_by_output(input_td);
            const auto parent = parent_expr->get_node();
            if (ov::is_type<op::Buffer>(parent) ||
                ov::is_type<op::VectorBuffer>(parent) ||
                ov::is_type<opset1::Parameter>(parent) ||
                ov::is_type<opset1::Constant>(parent))
                continue;

            const auto parent_port = parent_expr->get_output_port(input_td);
            const std::vector<TensorDescriptorPtr> parent_outs = { input_td };

            bool is_needed = ov::is_type<op::Brgemm>(parent);
            if (!is_needed) {
                // Find parent expr iterator inside the current Loop (Local iterator) - to check if the expr is in the Loop
                const auto local_parent_expr_iter = std::find(loop_begin_pos, loop_end_pos, parent_expr);
                // Find the nearest LoopEnd for parent - to figure out if parent expr is in another Loop.
                // It can be outside loop, then we don't need insert Buffer
                auto parent_loop_iter = std::find(linear_ir.cbegin(), loop_end_pos, parent_expr);
                while (parent_loop_iter != linear_ir.cend() &&
                       !ov::is_type<op::LoopBase>((*parent_loop_iter)->get_node())) {
                    parent_loop_iter = std::next(parent_loop_iter);
                }
                is_needed = local_parent_expr_iter == loop_end_pos && parent_loop_iter != linear_ir.cend() &&
                            ov::is_type<op::LoopEnd>((*parent_loop_iter)->get_node());
            }
            // If the parent expr is not in the current Loop but in another Loop (not outside Loop) or Brgemm, we insert Buffer
            if (is_needed) {
                auto buffer = std::make_shared<op::Buffer>(parent->output(parent_port), m_buffer_allocation_rank);

                const auto td = std::make_shared<TensorDescriptor>(input_td->get_tensor(),
                                                                   std::vector<size_t>{},  // or copy?
                                                                   input_td->get_layout());
                const std::vector<TensorDescriptorPtr> buffer_outs = { td };
                linear_ir.insert(loop_begin_pos, std::make_shared<LoweredExpr>(buffer, parent_outs, buffer_outs));
                linear_ir.replace_input(expr, input_td, td);
            }
        }

        // Insert Buffer after Loops if Loop contains nodes with consumers in another Loop or consumer is Brgemm
        for (const auto& output_td : expr->get_outputs()) {
            const auto out_port = expr->get_output_port(output_td);
            const std::vector<TensorDescriptorPtr> node_outs = {output_td};

            std::set<LoweredExprPtr> potential_consumers;
            std::set<LoweredExprPtr> buffers;
            const auto child_exprs = linear_ir.get_exprs_by_input(output_td);
            for (const auto &child_expr : child_exprs) {
                const auto child = child_expr->get_node();
                if (ov::is_type<op::Buffer>(child)) {
                    buffers.insert(child_expr);
                    continue;
                } else if (ov::is_type<op::Brgemm>(child)) {
                    potential_consumers.insert(child_expr);
                    continue;
                } else if (ov::is_type<opset1::Result>(child)) {
                    continue;
                }

                // Find child expr iterator inside the current Loop (Local iterator) - to check if the expr is in the Loop
                const auto local_child_expr_iter = std::find(loop_begin_pos, loop_end_pos, child_expr);
                // Find the nearest LoopBegin for child - to figure out if child expr is in another Loop.
                // It can be outside loop, then we don't need insert Buffer
                auto child_loop_iter = std::find(loop_begin_pos, linear_ir.cend(), child_expr);
                while (child_loop_iter != linear_ir.cbegin() && !ov::is_type<op::LoopBase>((*child_loop_iter)->get_node())) {
                    child_loop_iter = std::prev(child_loop_iter);
                }
                // If the child expr is not in the current Loop but in another Loop (not outside Loop) or Brgemm, we insert Buffer
                if (local_child_expr_iter == loop_end_pos &&
                    child_loop_iter != linear_ir.cbegin() &&
                    ov::is_type<op::LoopBegin>((*child_loop_iter)->get_node())) {
                    potential_consumers.insert(child_expr);
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

                auto buffer = std::make_shared<op::Buffer>(node->output(out_port), m_buffer_allocation_rank);
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
                linear_ir.insert(std::next(loop_end_pos),
                                 std::make_shared<LoweredExpr>(buffer, node_outs, buffer_outs));
                for (const auto consumer : potential_consumers) {
                    linear_ir.replace_input(consumer, output_td, td);
                }
            }
        }

        expr_it++;
    } while (expr_it != std::next(loop_end_pos) && expr_it != linear_ir.end());
}

bool InsertBuffer::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InsertBuffer")
    if (linear_ir.empty())
        return false;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        LoweredExprIR::constExprIt start, end;

        const auto& node = (*expr_it)->get_node();
        if (const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(node)) {
            const auto loop_end = loop_begin->get_loop_end();
            start = expr_it;
            end = std::find(start, linear_ir.cend(), linear_ir.get_expr_by_node(loop_end));
            OPENVINO_ASSERT(end != linear_ir.cend(), "LoopBegin must have corresponding LoopEnd in Linear IR!");
        } else if (const auto brgemm = ov::as_type_ptr<op::Brgemm>(node)) {
            start = expr_it;
            end = expr_it;
        } else {
            continue;
        }

        insertion(linear_ir, start, end);
        linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                            "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

