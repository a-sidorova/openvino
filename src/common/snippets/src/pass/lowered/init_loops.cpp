// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/init_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

namespace {
void get_io_exprs(LoweredExprIR& linear_ir, const std::vector<LoweredExprPort>& loop_entries, const std::vector<LoweredExprPort>& loop_exits,
                  std::vector<LoweredExprPtr>& loop_in_exprs, std::vector<LoweredExprPtr>& loop_out_exprs,
                  OutputVector& loop_in_outputs, OutputVector& loop_out_outputs) {
    loop_in_exprs.clear();
    loop_out_exprs.clear();
    loop_in_outputs.clear();
    loop_out_outputs.clear();

    for (const auto& loop_entry_point : loop_entries) {
        const auto expr = loop_entry_point.first;
        const auto node = expr->get_node();
        if (is_type<op::Load>(node) || is_type<op::BroadcastLoad>(node)) {
            // Todo: Sometimes several Load in one Loop read data from the same Node.
            if (std::none_of(loop_in_outputs.begin(), loop_in_outputs.end(),
                             [node](const ov::Output<ov::Node>& out) { return node->get_input_node_shared_ptr(0) == out.get_node_shared_ptr(); })) {
                loop_in_outputs.push_back(node->input_value(0));
                loop_in_exprs.push_back(expr);
            }
        }
    }

    for (const auto& loop_exit_point : loop_exits) {
        const auto expr = loop_exit_point.first;
        const auto node = expr->get_node();
        if (is_type<op::Store>(node)) {
            const auto &dest = node->output(0);
            loop_out_outputs.push_back(dest);
            loop_out_exprs.push_back(expr);
        }
    }
}

int64_t get_dim_stride(const size_t dim, const std::vector<size_t>& layout, const std::vector<size_t>& shape) {
    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim)
            break;
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
}  // namespace

InitLoops::InitLoops(size_t vector_size) : m_vector_size(vector_size), LinearIRTransformation() {}

std::vector<int64_t> InitLoops::init_ptr_increments(LoweredExprIR& linear_ir,
                                                    const std::vector<LoweredExprPtr>& loop_in_exprs,
                                                    const std::vector<LoweredExprPtr>& loop_out_exprs,
                                                    size_t dim_idx) const {
    std::vector<int64_t> ptr_increments;
    // Note: All loop inputs must have the same layout by definition.
    // If this doesn't hold, then we're trying to inject loops in the wrong place.
    const std::vector<size_t> loop_layout{
            !loop_in_exprs.empty() ? loop_in_exprs.front()->get_inputs()[0]->get_layout() :
            !loop_out_exprs.empty() ? loop_out_exprs.front()->get_outputs()[0]->get_layout() :
            std::vector<size_t>{}};
    // Note: Need to find max relevant dim first to account for broadcasting, collect relevant_dims as well
    size_t max_relevant_dim_size = 0;
    for (const auto& expr : loop_in_exprs) {
        const auto& out_tds = expr->get_outputs();
        const auto& dst_layout = out_tds[0]->get_layout();
        const auto& dst_tensor = out_tds[0]->get_tensor();
        const auto& dst_dim = *(dst_layout.rbegin() + dim_idx);
        max_relevant_dim_size = std::max(dst_tensor[dst_dim], max_relevant_dim_size);
        if (loop_layout != expr->get_inputs()[0]->get_layout())
            throw ngraph_error("InitLoops noticed an attempt to init loop with inconsistent input layouts");
    }
    for (const auto& expr : loop_out_exprs) {
        const auto& out_tds = expr->get_outputs();
        const auto& dst_layout = out_tds[0]->get_layout();
        const auto& dst_tensor = out_tds[0]->get_tensor();
        const auto& dst_dim = *(dst_layout.rbegin() + dim_idx);
        max_relevant_dim_size = std::max(dst_tensor[dst_dim], max_relevant_dim_size);
        if (loop_layout != expr->get_outputs()[0]->get_layout())
            throw ngraph_error("InitLoops noticed an attempt to init loop with inconsistent input layouts");
    }
    for (const auto& expr : loop_in_exprs) {
        const auto& out_tds = expr->get_outputs();
        const auto& src_tensor = expr->get_inputs().front()->get_tensor();
        const auto& dst_layout = out_tds[0]->get_layout();
        const auto& dst_dim = *(dst_layout.rbegin() + dim_idx);
        int64_t ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (!(src_tensor[dst_dim] == 1 && max_relevant_dim_size != 1))
            ptr_increment = get_dim_stride(dst_dim, loop_layout, src_tensor);
        ptr_increments.push_back(ptr_increment);
    }
    // Note: Le already accounted for loop_input vs inside loops layout mismatch. So we need non-dense output
    // ptr_increments only if loop_input_layout doesn't match loop_output_layout
    for (const auto& expr : loop_out_exprs) {
        const auto& out_tds = expr->get_outputs();
        const auto& dst_layout = out_tds[0]->get_layout();
        const auto& dst_tensor = out_tds[0]->get_tensor();
        const auto& dst_dim = *(loop_layout.rbegin() + dim_idx);
        int64_t ptr_increment = 0;
        // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
        if (!(dst_tensor[dst_dim] == 1 && max_relevant_dim_size != 1))
            ptr_increment = get_dim_stride(dst_dim, dst_layout, dst_tensor);
        ptr_increments.push_back(ptr_increment);
    }

    return ptr_increments;
}

std::vector<int64_t> InitLoops::init_finalization_offsets(const std::vector<int64_t>& ptr_increments, size_t work_amount) const {
    std::vector<int64_t> finalization_offsets;
    for (const auto& ptr_incr : ptr_increments) {
        int64_t offset = -1 * ptr_incr * work_amount;
        finalization_offsets.push_back(offset);
    }
    return finalization_offsets;
}

void InitLoops::insertion(LoweredExprIR& linear_ir, const LoweredLoopManager::LoweredLoopInfoPtr& loop_info, size_t dim_idx, bool& has_outer_loop) {
    const auto loop_entries = loop_info->m_entry_exprs;
    const auto loop_exits = loop_info->m_exit_exprs;
    const auto work_amount = loop_info->m_work_amount;
    const auto work_amount_increment = loop_info->m_increment;

    OutputVector loop_in_outputs, loop_out_outputs;
    std::vector<LoweredExprPtr> loop_in_exprs, loop_out_exprs;
    get_io_exprs(linear_ir, loop_entries, loop_exits,
                 loop_in_exprs, loop_out_exprs,
                 loop_in_outputs, loop_out_outputs);

    // If we don't need Loop (explicit execution without cycles), we don't explicitly insert the Loop expressions
    const auto subtensor_in = loop_in_exprs.empty() ? std::vector<size_t>{} : loop_in_exprs.front()->get_inputs().front()->get_subtensor();
    const bool explicit_execution = work_amount == 0 || dim_idx == 0 && subtensor_in.size() > 1 && subtensor_in.back() == work_amount;
    if (explicit_execution) {
        has_outer_loop |= false;
        return;
    }

    LoweredExprIR::constExprIt loop_begin_pos, loop_end_pos;
    LoweredLoopManager::get_loop_bounds(linear_ir, loop_entries, loop_exits, loop_begin_pos, loop_end_pos);

    auto ptr_increments = init_ptr_increments(linear_ir, loop_in_exprs, loop_out_exprs, dim_idx);
    auto finalization_offsets = init_finalization_offsets(ptr_increments, work_amount);

    const auto& loop_begin = std::make_shared<op::LoopBegin>();
    const auto& loop_begin_expr = std::make_shared<LoweredExpr>(loop_begin, std::vector<TensorDescriptorPtr>{});
    linear_ir.insert(loop_begin_pos, loop_begin_expr);

    OutputVector managed_outputs = loop_in_outputs;
    managed_outputs.insert(managed_outputs.end(), loop_out_outputs.begin(), loop_out_outputs.end());
    managed_outputs.push_back(loop_begin->output(0));

    auto is_buffer_input = [&linear_ir](const LoweredExprPtr& expr) {
        const auto parent_expr = linear_ir.get_expr_by_output(expr->get_inputs().front());
        return ov::is_type<op::Buffer>(parent_expr->get_node());
    };
    auto is_buffer_output = [&linear_ir](const LoweredExprPtr& expr) {
        const auto child_exprs = linear_ir.get_exprs_by_input(expr->get_outputs().front());
        return ov::is_type<op::Buffer>((*child_exprs.begin())->get_node());
    };
    auto there_is_buffer = std::any_of(loop_in_exprs.begin(), loop_in_exprs.end(), is_buffer_input) ||
                           std::any_of(loop_out_exprs.begin(), loop_out_exprs.end(), is_buffer_output);

    const auto& loop_end = std::make_shared<op::LoopEnd>(
            managed_outputs, work_amount, work_amount_increment, ptr_increments, finalization_offsets);
    loop_end->set_work_with_buffer(there_is_buffer);
    loop_end->has_outer_loop = has_outer_loop;
    // set internal flag to enable scalar vs vector loop optimizations
    // loop_end->has_outer_loop = has_outer_loop;

    std::vector<TensorDescriptorPtr> loop_end_inputs;
    for (const auto& expr : loop_in_exprs)
        loop_end_inputs.push_back(expr->get_inputs().front());
    for (const auto& expr : loop_out_exprs)
        loop_end_inputs.push_back(expr->get_outputs().front());
    loop_end_inputs.push_back(linear_ir.get_expr_by_node(loop_begin)->get_outputs().front());

    const auto& loop_end_expr = std::make_shared<LoweredExpr>(loop_end, loop_end_inputs);
    linear_ir.insert(loop_end_pos, loop_end_expr);
    has_outer_loop |= true;
    return;
}

bool InitLoops::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;
    const auto& lowering_config = linear_ir.get_config();
    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> inserted_loops;
    std::stack<size_t> current_loops;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopBase>(node) ||
            ov::is_type<op::Buffer>(node) ||     // Need to cover Buffer
            ov::is_type<opset1::Parameter>(node) ||
            ov::is_type<opset1::Result>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto expr_loops = expr->get_loop_ids();
        const auto loop_depth = expr_loops.size();
        bool has_outer_loop = false;
        for (size_t i = 0; i < loop_depth; ++i) {
            const auto loop_id = expr_loops[i];
            bool need_to_insert = inserted_loops.find(loop_id) == inserted_loops.end();
            if (need_to_insert) {
                const auto loop_info = loop_manager->get(loop_id);
                insertion(linear_ir, loop_info, loop_depth - i - 1, has_outer_loop);
                inserted_loops.insert(loop_id);  // save Loop ID
                linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                                    "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
            }
        }
    }

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

