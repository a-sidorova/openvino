// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using LoopPort = LinearIR::LoopManager::LoopPort;

namespace {
int64_t get_dim_stride(const LinearIR::LoopManagerPtr& loop_manager, const std::vector<size_t>& loop_ids,
                       size_t loop_id, size_t dim, size_t dim_idx,
                       const std::vector<size_t>& layout, const std::vector<size_t>& shape, bool is_tail = false) {
    // Example, shape = [3, 384, 64], loop_ids = [2, 1, 0], layout = [0, 1, 2]
    // Loop Info:                            | Pointer increments:
    // - 2: work_amount = 12, dim_idx = 1    | 1 x 64 x 32 - (1 * shape[layout[2]] * 32) where 32 is work_amount of inner splitted Loop
    // - 1: work_amount = 32, dim_idx = 1    | 1 x 64 - (1 * shape[layout[2]])
    // - 0: work_amount = 64, dim_idx = 0    | 1
    // Note that dim_idx enumerates dimensions from the end: 64, 384, 3
    // Firstly, we find all Loop IDs with the same dimension index.
    // The Loop Info's with the same dimension index mean that these Loops split this dimension together.
    // It's possible in Brgemm Blocking by M, for example
    std::vector<size_t> splitted_loops;
    // Inner -> Outer
    for (auto it = loop_ids.rbegin(); it != loop_ids.rend(); ++it) {
        const auto id = *it;
        if (loop_manager->get_loop_info(id)->dim_idx == dim_idx) {
            splitted_loops.push_back(id);
        }
    }

    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim) {
            // We added work amount of inner splitted Loops
            for (auto id : splitted_loops) {
                if (id == loop_id)
                    break;
                const auto loop_info = is_tail ? loop_manager->get_loop_info(id)->tail_info : loop_manager->get_loop_info(id);
                OPENVINO_ASSERT(loop_info != nullptr, "LoopInfo has not been found!");
                stride *= loop_info->work_amount;
            }
            break;
        }
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
}  // namespace

InitLoops::InitLoops() : Pass() {}

void InitLoops::init(const LinearIR::LoopManagerPtr& loop_manager, const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id, bool is_tail) {
    // the exact number of iterations without remainder
    //const auto iter_count = static_cast<size_t>(loop_info->work_amount / loop_info->increment) * loop_info->increment;
    const auto dim_idx = loop_info->dim_idx;

    init_ptr_increments(loop_info->entry_points, loop_info->exit_points, loop_manager, loop_id, dim_idx, is_tail);
    init_finalization_offsets(loop_info->entry_points, loop_info->exit_points, loop_info->work_amount);
    init_element_type_sizes(loop_info->entry_points, loop_info->exit_points);
}

void InitLoops::init_ptr_increments(std::vector<LoopPort>& loop_inputs, std::vector<LoopPort>& loop_outputs,
                                    const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id, size_t dim_idx, bool is_tail) {
    // Find dimension to figure out there is broadcasting by this input/output or not for each loop port
    size_t broadcastable_dim = 1;
    for (auto& loop_input : loop_inputs) {
        const auto& port = loop_input.expr_port;
        // For strides we have to use layout from source since source writes data by special rules
        const auto& layout = port->get_descriptor_ptr()->get_layout();
        const auto& shape = port->get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        broadcastable_dim = std::max(broadcastable_dim, shape[dim]);
    }

    for (auto& loop_output : loop_outputs) {
        const auto& port = loop_output.expr_port;
        const auto& layout = port->get_descriptor_ptr()->get_layout();
        const auto& shape = port->get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        broadcastable_dim = std::max(broadcastable_dim, shape[dim]);
    }

    for (auto& loop_input : loop_inputs) {
        const auto& port = loop_input.expr_port;
        // For strides we have to use layout from source since source writes data by special rules
        const auto source = *port->get_connected_ports().begin();
        const auto loop_ids = port->get_expr()->get_loop_ids();
        const auto& layout = port->get_descriptor_ptr()->get_layout();
        const auto& shape = port->get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        loop_input.ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (loop_input.is_incremented && !(shape[dim] == 1 && broadcastable_dim != 1)) {
            loop_input.ptr_increment = get_dim_stride(loop_manager, loop_ids, loop_id, dim, dim_idx,
                                                      source.get_descriptor_ptr()->get_layout(), shape, is_tail);
        }
    }

    for (auto& loop_output : loop_outputs) {
        const auto& port = loop_output.expr_port;
        const auto loop_ids = port->get_expr()->get_loop_ids();
        const auto& layout = port->get_descriptor_ptr()->get_layout();
        const auto& shape = port->get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        // [113106] Need to update order logic
        std::vector<size_t> planar_layout(layout.size());
        std::iota(planar_layout.begin(), planar_layout.end(), 0);
        loop_output.ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (loop_output.is_incremented && !(shape[dim] == 1 && broadcastable_dim != 1)) {
            loop_output.ptr_increment = get_dim_stride(loop_manager, loop_ids, loop_id, dim, dim_idx,
                                                       planar_layout, shape, is_tail);
        }
    }
}

void InitLoops::init_finalization_offsets(std::vector<LinearIR::LoopManager::LoopPort>& loop_inputs,
                                          std::vector<LinearIR::LoopManager::LoopPort>& loop_outputs,
                                          size_t work_amount) {
    for (auto& loop_input : loop_inputs) {
        loop_input.finalization_offset = -1 * loop_input.ptr_increment * work_amount;
    }
    for (auto& loop_output : loop_outputs) {
        loop_output.finalization_offset = -1 * loop_output.ptr_increment * work_amount;
    }
}

void InitLoops::init_element_type_sizes(std::vector<LoopPort>& loop_inputs,
                                        std::vector<LoopPort>& loop_outputs) {
    for (auto& loop_input : loop_inputs) {
        const auto& port = loop_input.expr_port;
        loop_input.data_size = static_cast<int64_t>(port->get_expr()->get_node()->get_input_element_type(port->get_index()).size());
    }
    for (auto& loop_output : loop_outputs) {
        const auto& port = loop_output.expr_port;
        loop_output.data_size = static_cast<int64_t>(port->get_expr()->get_node()->get_output_element_type(port->get_index()).size());
    }
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();
    for (const auto& loop : loops) {
        const auto loop_id = loop.first;
        const auto loop_info = loop.second;

        init(loop_manager, loop_info, loop_id);
        if (loop_info->tail_info)
            init(loop_manager, loop_info->tail_info, loop_id, true);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
