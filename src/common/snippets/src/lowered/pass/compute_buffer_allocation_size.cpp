// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/compute_buffer_allocation_size.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
size_t calculate_size(const VectorDims& shape) {
    size_t size = 1;
    for (const auto& v : shape) {
        if (utils::is_dynamic_value(v))
            return utils::get_dynamic_value<size_t>();
        size *= v;
    }
    return size;
}
}  // namespace

// Ticket: 113744
// TODO: This logic covers only several specific cases so it should be generalized.
size_t ComputeBufferAllocationSize::get_allocation_size(const LinearIR::LoopManagerPtr& loop_manager,
                                                        const ExpressionPtr& buffer_expr, size_t allocation_rank) {
    const auto& parent_port = buffer_expr->get_input_port_connector(0)->get_source();
    const auto& parent_loop_ids = parent_port.get_expr()->get_loop_ids();
    const auto& buffer_loop_ids = buffer_expr->get_loop_ids();
    const auto planar_shape = utils::get_preordered_vdims(parent_port);

    const size_t rank = allocation_rank >= 0 ? std::min(static_cast<size_t>(allocation_rank), planar_shape.size()) : planar_shape.size();
    VectorDims allocation_shape(planar_shape.cbegin() + (planar_shape.size() - rank), planar_shape.cend());

    if (buffer_loop_ids.empty() || parent_loop_ids.empty()) {
        return calculate_size(allocation_shape);
    }

    // If subtensor is set, its information is used for allocation shape computation. Two situations are possible:
    // 1. Buffer is outside the parent loop: the corresponding subtensor value is ignored, parent loop work amount is set instead
    // 2. Buffer is inside the parent loop: the corresponding subtensor value is used in allocation shape.
    // Since we can defenitely know which subtensor value corresponds to the loop only for 1st case
    // (we can extract this info from loop exit port), we copy subtensor, and then replace subtensor values with parent loop work amount if needed.
    // Example:
    // Parent subtensor: [M_blk, N_blk]
    // Buffer loop idces: [M_loop_idx], parent loop idces: [M_loop_idx, N_loop_idx]
    //
    // 1. Allocation shape is set to subtensor: [M_blk, N_blk]
    // 2. Buffer is inside M_loop_idx loop => allocation shape is not changed
    // 3. Buffer is outside N_loop_idx loop => the corresponding allocation shape value is replaced with N loop work amount
    // So the result allocation shape is [M_blk, N_loop_work_amount]
    const auto& subtensor = parent_port.get_descriptor_ptr()->get_subtensor();
    if (!subtensor.empty()) {
        for (size_t i = 0; i < std::min(rank, subtensor.size()); ++i) {
            auto& cur_val = *(allocation_shape.rbegin() + i);
            const auto& subtensor_val = *(subtensor.rbegin() + i);
            cur_val = std::min(cur_val, subtensor_val);
        }
        for (const auto& parent_loop : parent_loop_ids) {
            if (std::find(buffer_loop_ids.begin(), buffer_loop_ids.end(), parent_loop) == buffer_loop_ids.end()) {
                const auto loop_info = loop_manager->get_loop_info(parent_loop);
                const auto& exit_points = loop_info->get_exit_points();
                auto it = std::find_if(exit_points.begin(),
                                       exit_points.end(),
                                       [&parent_port](const LinearIR::LoopManager::LoopPort& port) {
                                           return *port.expr_port == parent_port;
                                       });
                OPENVINO_ASSERT(it != exit_points.end(), "compute_allocation_shape: exit point of parent loop can not be found");
                const auto& loop_port = *it;
                if (loop_port.is_incremented && loop_port.dim_idx < allocation_shape.size()) {
                    *(allocation_shape.rbegin() + loop_port.dim_idx) = loop_info->get_work_amount();
                }
            }
        }
    } else {
        // WA: In case of empty subtensors another information have to be used to update allocation shape.
        for (size_t i = 0; i < std::min(rank, parent_loop_ids.size()); ++i) {
            const auto loop = loop_manager->get_loop_info(*(parent_loop_ids.rbegin() + i));
            OPENVINO_ASSERT(loop->get_dim_idx() == i, "compute_allocation_shape: eltwise loop has unexpected dimension index");
            *(allocation_shape.rbegin() + i) = loop->get_work_amount();
        }
        if (allocation_rank > parent_loop_ids.size()) {
            for (size_t i = 0; i < allocation_rank - parent_loop_ids.size(); ++i) {
                allocation_shape[i] = 1;
            }
        }
    }
    return calculate_size(allocation_shape);
}

bool ComputeBufferAllocationSize::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ComputeBufferAllocationSize")

    const auto& loop_manager = linear_ir.get_loop_manager();

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        const auto node = (*expr_it)->get_node();
        if (const auto buffer = ov::as_type_ptr<op::IntermediateMemoryBuffer>(node)) {
            buffer->set_allocation_size(get_allocation_size(loop_manager, expr, m_buffer_allocation_rank));
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
