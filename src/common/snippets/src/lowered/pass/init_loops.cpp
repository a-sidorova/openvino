// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using LoopPort = LinearIR::LoopManager::LoopPort;

namespace {
int64_t get_stride(size_t dim, const VectorDims& shape) {
    int64_t stride = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        stride *= static_cast<int64_t>(shape[i]);
    }
    return stride;
}
}  // namespace

void InitLoops::init_loop_info(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    const auto work_amount = loop_info->get_work_amount();

    auto init_args = [&](std::vector<LoopPort>& loop_ports) {
        for (auto& loop_port : loop_ports) {
            init_ptr_increment(loop_port, work_amount);
            init_finalization_offset(loop_port, work_amount);
            init_data_size(loop_port);
        }
    };

    auto entry_points = loop_info->get_entry_points();
    auto exit_points = loop_info->get_exit_points();
    init_args(entry_points);
    init_args(exit_points);
    loop_info->set_entry_points(entry_points);
    loop_info->set_exit_points(exit_points);
}

void InitLoops::init_ptr_increment(LinearIR::LoopManager::LoopPort& loop_port, size_t work_amount) {
    loop_port.ptr_increment = 0;
    if (!loop_port.is_incremented)
        return;

    const auto& expr_port = loop_port.expr_port;
    const auto& shape = expr_port->get_descriptor_ptr()->get_shape();
    size_t dim = loop_port.dim_idx;
    if (const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr_port->get_expr()->get_node())) {
        if (expr_port->get_type() == ExpressionPort::Input)
            dim = utils::get_input_dim_idx(ma->get_input_order(expr_port->get_index()), loop_port.dim_idx);
        else if (expr_port->get_type() == ExpressionPort::Output)
            dim = utils::get_output_dim_idx(ma->get_output_order(expr_port->get_index()), loop_port.dim_idx);
        else
            OPENVINO_THROW("Unsupported expression port type!");
    }
    // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
    if (!(shape[dim] == 1 && work_amount != 1)) {
        loop_port.ptr_increment = get_stride(dim, shape);
    }
}

void InitLoops::init_finalization_offset(LinearIR::LoopManager::LoopPort& loop_port, size_t work_amount) {
    loop_port.finalization_offset = -1 * loop_port.ptr_increment * work_amount;
}

void InitLoops::init_data_size(LinearIR::LoopManager::LoopPort& loop_port) {
    const auto& expr_port = loop_port.expr_port;
    if (expr_port->get_type() == ExpressionPort::Input) {
        loop_port.data_size = static_cast<int64_t>(expr_port->get_expr()->get_node()->get_input_element_type(expr_port->get_index()).size());
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        loop_port.data_size = static_cast<int64_t>(expr_port->get_expr()->get_node()->get_output_element_type(expr_port->get_index()).size());
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();
    for (const auto& loop : loops) {
        init_loop_info(loop.second);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
