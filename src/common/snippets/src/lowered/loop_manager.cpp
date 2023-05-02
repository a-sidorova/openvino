// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_manager.hpp"

#include "snippets/lowered/expression.hpp"
#include "snippets/utils.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/type.hpp>

#include <snippets/itt.hpp>

namespace ngraph {
namespace snippets {
namespace lowered {

size_t LinearIR::LoopManager::add_loop_info(const LoopInfoPtr &loop) {
    const auto index = next_id;
    m_map[index] = loop;
    next_id++;
    return index;
}

void LinearIR::LoopManager::remove_loop_info(size_t index) {
    m_map.erase(index);
}

using LoopInfoPtr = LinearIR::LoopManager::LoopInfoPtr;

const std::map<size_t, LoopInfoPtr> &LinearIR::LoopManager::get_map() const {
    return m_map;
}

LoopInfoPtr LinearIR::LoopManager::get_loop_info(size_t index) const {
    const auto it = m_map.find(index);
    OPENVINO_ASSERT(it != m_map.end(), "LoopInformation hasn't been found!");
    return it->second;
}

void LinearIR::LoopManager::get_loop_bounds(const LinearIR &linear_ir,
                                            size_t loop_id,
                                            LinearIR::constExprIt &loop_begin_pos,
                                            LinearIR::constExprIt &loop_end_pos) const {
    const auto loop_info = get_loop_info(loop_id);
    get_loop_bounds(linear_ir, loop_info->entry_exprs, loop_info->exit_exprs, loop_begin_pos, loop_end_pos, loop_id);
}

void LinearIR::LoopManager::get_loop_bounds(const LinearIR &linear_ir,
                                            const std::vector<TensorDescriptor> &entries,
                                            const std::vector<TensorDescriptor> &exits,
                                            LinearIR::constExprIt &loop_begin_pos,
                                            LinearIR::constExprIt &loop_end_pos,
                                            size_t loop_id) {
    OPENVINO_ASSERT(!entries.empty(), "Loop must have entry points");
    OPENVINO_ASSERT(!exits.empty(), "Loop must have entry points");
    const auto& entry_expr = entries.front().get_expr_ptr();
    loop_begin_pos = std::find(linear_ir.begin(), linear_ir.end(), entry_expr);
    OPENVINO_ASSERT(loop_begin_pos != linear_ir.end(), "Loop begin hasn't been found!");

    // Some operations in Loop can be before first entry points: Scalars, VectorBuffer.
    // We should iterate by them till the expr is in the corresponding Loop
    auto prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    while (std::find(prev_loop_ids.begin(), prev_loop_ids.end(), loop_id) != prev_loop_ids.end()) {
        loop_begin_pos = std::prev(loop_begin_pos);
        prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    }

    // At the moment all Loops must have exit points
    const auto& exit_expr = exits.back().get_expr_ptr();
    loop_end_pos = std::next(std::find(loop_begin_pos, linear_ir.end(), exit_expr));
    OPENVINO_ASSERT(loop_end_pos != linear_ir.end(), "Loop end hasn't been found!");
}

void LinearIR::LoopManager::get_io_loop_ports(LinearIR::constExprIt loop_begin_pos,
                                              LinearIR::constExprIt loop_end_pos,
                                              std::vector<TensorDescriptor> &entries,
                                              std::vector<TensorDescriptor> &exits) {
    entries.clear();
    exits.clear();
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        const auto& expr = *expr_it;
        const auto inputs = expr->get_inputs();
        const auto outputs = expr->get_outputs();

        for (size_t in_port = 0; in_port < inputs.size(); ++in_port) {
            const auto in_td = inputs[in_port];
            const auto parent_expr = in_td->get_source().get_expr_ptr();
            if (!ov::is_type<ov::op::v0::Constant>(parent_expr->get_node()) &&
                std::find(loop_begin_pos, expr_it, parent_expr) == expr_it) {
                entries.push_back(expr->input_port(in_port));
            }
        }

        for (size_t out_port = 0; out_port < outputs.size(); ++out_port) {
            const auto out_td = outputs[out_port];
            const auto consumer_ports = out_td->get_consumers();
            for (const auto& consumer : consumer_ports) {
                const auto consumer_expr = consumer.get_expr_ptr();
                if (std::find(expr_it, loop_end_pos, consumer_expr) == loop_end_pos) {
                    exits.push_back(expr->output_port(out_port));
                    break;
                }
            }
        }
    }
}

void LinearIR::LoopManager::skipped_mark(LinearIR::constExprIt loop_begin_pos,
                                         LinearIR::constExprIt loop_end_pos,
                                         size_t loop_depth) {
    const auto loop_ids = std::vector<size_t>(loop_depth, Expression::LOOP_NULL_ID);
    for (auto& expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        const auto expr = *expr_it;
        expr->set_loop_ids(loop_ids);
    }
}

void LinearIR::LoopManager::mark_loop(LinearIR::constExprIt loop_begin_pos,
                                      LinearIR::constExprIt loop_end_pos,
                                      size_t loop_depth, size_t vector_size) {
    std::vector<TensorDescriptor> loop_entry_points, loop_exit_points;
    LoopManager::get_io_loop_ports(loop_begin_pos, loop_end_pos, loop_entry_points, loop_exit_points);

    auto broadcast = [](std::vector<size_t> &lhs, const std::vector<size_t> &rhs) -> void {
        if (rhs == lhs)
            return;
        const auto lhs_size = lhs.size();
        const auto rhs_size = rhs.size();
        const auto size = std::max(lhs_size, rhs_size);
        lhs.resize(size, 1);
        for (size_t i = 0; i < size; ++i) {
            const auto lhs_value = i < lhs_size ? *(lhs.crbegin() + i) : 1;
            const auto rhs_value = i < rhs_size ? *(rhs.crbegin() + i) : 1;
            OPENVINO_ASSERT(lhs_value == rhs_value || lhs_value == 1 || rhs_value == 1,
                            "Output shapes of Loop must be broadcastable!");
            *(lhs.rbegin() + i) = std::max(lhs_value, rhs_value);
        }
    };

    auto found_port = [](const std::vector<TensorDescriptor>& ports, const TensorDescriptor& target) {
        return std::find_if(ports.begin(), ports.end(), [&target](const TensorDescriptor& port) {
            return port.get_expr_ptr().get() == target.get_expr_ptr().get() &&
                   port.get_index() == target.get_index() &&
                   port.get_type() == target.get_type();
        }) != ports.end();
    };

    std::vector<size_t> loop_subtensor;
    std::vector<size_t> loop_layout;
    std::vector<size_t> loop_tensor(1, 1);  // Scalar
    for (const auto& exit_point : loop_exit_points) {
        const auto out_tensor = utils::get_reordered_shape(exit_point.get_tensor(), exit_point.get_layout());
        broadcast(loop_tensor, out_tensor);

        // SubTensor and Layout inside Loops must be the same.
        // We have to verify that input of exit point isn't entry point or Constant to check for subtensor and layout because of
        // then this input is not inside Loop
        const auto& expr = exit_point.get_expr_ptr();
        for (size_t i = 0; i < expr->get_input_count(); ++i) {
            const auto port = expr->input_port(i);
            const auto parent = expr->get_inputs()[port.get_index()]->get_source().get_expr_ptr()->get_node();
            if (!found_port(loop_entry_points, port) && !ov::is_type<ov::op::v0::Constant>(parent)) {
                if (loop_subtensor.empty())
                    loop_subtensor = port.get_subtensor();
                if (loop_layout.empty())
                    loop_layout = port.get_layout();
                OPENVINO_ASSERT(loop_subtensor == port.get_subtensor(), "SubTensor inside Loop must be the same");
                OPENVINO_ASSERT(loop_layout == port.get_layout(), "Layout inside Loop must be the same");
            }
        }
    }

    for (const auto& entry_point : loop_entry_points) {
        const auto in_tensor = utils::get_reordered_shape(entry_point.get_tensor(), entry_point.get_layout());
        broadcast(loop_tensor, in_tensor);

        // SubTensor and Layout inside Loops must be the same.
        // We have to verify that output of entry point isn't exit point to check for subtensor and layout because of
        // then this output is not inside Loop
        const auto& expr = entry_point.get_expr_ptr();
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto port = expr->output_port(i);
            if (!found_port(loop_exit_points, port)) {
                if (loop_subtensor.empty())
                    loop_subtensor = port.get_subtensor();
                if (loop_layout.empty())
                    loop_layout = port.get_layout();
                OPENVINO_ASSERT(loop_subtensor == port.get_subtensor(), "SubTensor inside Loop must be the same");
                OPENVINO_ASSERT(loop_layout == port.get_layout(), "Layout inside Loop must be the same");
            }
        }
    }

    for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
        OPENVINO_ASSERT(dim_idx < loop_tensor.size(), "Incorrect indexes of Loop for markup");
        const auto work_amount =
                loop_tensor.size() > dim_idx ? *(loop_tensor.rbegin() + dim_idx)
                                             : 0;
        const auto work_amount_increment =
                loop_subtensor.size() > dim_idx ? *(loop_subtensor.rbegin() + dim_idx)
                                                : (dim_idx == 0 ? vector_size : 1);

        mark_loop(loop_begin_pos, loop_end_pos, loop_depth - dim_idx - 1, work_amount,
                  work_amount_increment, loop_entry_points, loop_exit_points);
    }
}

void LinearIR::LoopManager::mark_loop(LinearIR::constExprIt loop_begin_pos,
                                      LinearIR::constExprIt loop_end_pos,
                                      size_t idx,
                                      size_t work_amount,
                                      size_t work_amount_increment,
                                      const std::vector<TensorDescriptor> &entries,
                                      const std::vector<TensorDescriptor> &exits) {
    const auto loop_info = std::make_shared<LoopManager::LoopInfo>(work_amount, work_amount_increment, entries, exits);
    const auto loop_id = this->add_loop_info(loop_info);
    exprs_marking(loop_begin_pos, loop_end_pos, loop_id, idx);
}

void LinearIR::LoopManager::exprs_marking(LinearIR::constExprIt loop_begin_pos,
                                          LinearIR::constExprIt loop_end_pos,
                                          size_t loop_id, size_t idx) {
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        expr_it->get()->set_loop_id(loop_id, idx);
    }
}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
