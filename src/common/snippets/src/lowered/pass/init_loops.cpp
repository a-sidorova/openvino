// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
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
        if (utils::is_dynamic_vdim(shape[i])) {
            return LoopPort::DYNAMIC_VALUE;
        }
        stride *= static_cast<int64_t>(shape[i]);
    }
    return stride;
}
}  // namespace

void InitLoops::init_loop_info(const LinearIR::LoopManager::LoopInfoPtr& loop_info, bool only_runtime_args) {
    init_work_amount(loop_info);

    const auto work_amount = loop_info->get_work_amount();

    auto init_args = [&](std::vector<LoopPort>& loop_ports) {
        for (auto& loop_port : loop_ports) {
            init_ptr_increment(loop_port, work_amount);
            init_finalization_offset(loop_port, work_amount);
            if (!only_runtime_args)
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

void InitLoops::init_ptr_increment(LoopPort& loop_port, size_t work_amount) {
    loop_port.ptr_increment = 0;
    if (!loop_port.is_incremented)
        return;

    const auto& expr_port = loop_port.expr_port;
    const auto& layout = expr_port->get_descriptor_ptr()->get_layout();
    const auto& shape = expr_port->get_descriptor_ptr()->get_shape();
    if (expr_port->get_type() == ExpressionPort::Input) {
        const auto& dim = utils::get_input_dim_idx(layout, layout.size() - 1 - loop_port.dim_idx);
        // When we cannot say about broadcasting by last dim
        if (dim == shape.size() - 1 && utils::is_dynamic_vdim(shape[dim])) {
            loop_port.ptr_increment = LoopPort::DYNAMIC_VALUE;
        } else if (!(shape[dim] == 1 && work_amount != 1)) {
            loop_port.ptr_increment = get_stride(dim, shape);
        }
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        const auto& dim = utils::get_output_dim_idx(layout, layout.size() - 1 - loop_port.dim_idx);
        // When we cannot say about broadcasting by last dim
        if (dim == shape.size() - 1 && utils::is_dynamic_vdim(shape[dim])) {
            loop_port.ptr_increment = LoopPort::DYNAMIC_VALUE;
        } else if (!(shape[dim] == 1 && work_amount != 1)) {
            // Output layout shows how we already written data by which order and strides
            loop_port.ptr_increment = get_stride(dim, shape);
        }
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
}

void InitLoops::init_finalization_offset(LoopPort& loop_port, size_t work_amount) {
    loop_port.finalization_offset =
        utils::is_dynamic_vdim(work_amount) || LoopPort::is_dynamic_value(loop_port.ptr_increment) ? LoopPort::DYNAMIC_VALUE
                                                                                                   : -1 * loop_port.ptr_increment * work_amount;
}

void InitLoops::init_data_size(LoopPort& loop_port) {
    const auto& expr_port = loop_port.expr_port;
    if (expr_port->get_type() == ExpressionPort::Input) {
        loop_port.data_size = static_cast<int64_t>(expr_port->get_expr()->get_node()->get_input_element_type(expr_port->get_index()).size());
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        loop_port.data_size = static_cast<int64_t>(expr_port->get_expr()->get_node()->get_output_element_type(expr_port->get_index()).size());
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
}

void InitLoops::init_work_amount(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    if (!utils::is_dynamic_vdim(loop_info->get_work_amount()))
        return;

    auto broadcast = [](size_t& lhs_value, const size_t& rhs_value) -> void {
        if (lhs_value == rhs_value || lhs_value == 1 || utils::is_dynamic_vdim(rhs_value)) {
            lhs_value = rhs_value;
            return;
        } else if (rhs_value == 1 || utils::is_dynamic_vdim(lhs_value)) {
            return;
        }
        OPENVINO_THROW("Dimensions of shapes aren't broadcastable for work amount initialization!");
    };

    size_t work_amount = 1;
    for (const auto& loop_port : loop_info->get_entry_points()) {
        if (loop_port.is_incremented) {
            const auto& shape = loop_port.expr_port->get_descriptor_ptr()->get_shape();
            const auto& layout = loop_port.expr_port->get_descriptor_ptr()->get_layout();
            broadcast(work_amount, shape[utils::get_input_dim_idx(layout, layout.size() - 1 - loop_port.dim_idx)]);
        }
    }
    for (const auto& loop_port : loop_info->get_exit_points()) {
        if (loop_port.is_incremented) {
            const auto& shape = loop_port.expr_port->get_descriptor_ptr()->get_shape();
            const auto& layout = loop_port.expr_port->get_descriptor_ptr()->get_layout();
            broadcast(work_amount, shape[utils::get_output_dim_idx(layout, layout.size() - 1 - loop_port.dim_idx)]);
        }
    }
    loop_info->set_work_amount(work_amount);
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
