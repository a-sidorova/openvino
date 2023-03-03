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
void get_io_exprs(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt begin, LoweredExprIR::constExprIt end,
                  std::vector<LoweredExprPtr>& loop_in_exprs, std::vector<LoweredExprPtr>& loop_out_exprs,
                  OutputVector& loop_in_outputs, OutputVector& loop_out_outputs) {
    loop_in_exprs.clear();
    loop_out_exprs.clear();
    loop_in_outputs.clear();
    loop_out_outputs.clear();
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& node = (*expr_it)->get_node();
        if (is_type<op::Load>(node) || is_type<op::BroadcastLoad>(node)) {
            const auto parent_expr = linear_ir.get_expr_by_output((*expr_it)->get_inputs().front());
            // We should increment pointers of IO expr which are outside of Loop
            // Note: Sometimes several Load in one Loop read data from the same Node.
            if (std::find(begin, end, parent_expr) == end &&
                std::none_of(loop_in_outputs.begin(), loop_in_outputs.end(),
                             [node](const ov::Output<ov::Node>& out) { return node->get_input_node_shared_ptr(0) == out.get_node_shared_ptr(); })) {
                loop_in_outputs.push_back(node->input_value(0));
                loop_in_exprs.push_back(*expr_it);
            }
        } else if (is_type<op::Store>(node)) {
            const auto consumer_exprs = linear_ir.get_exprs_by_input((*expr_it)->get_outputs().front());
            for (const auto& consumer_expr : consumer_exprs) {
                // We should increment pointers of IO expr which are outside of Loop
                if (std::find(begin, end, consumer_expr) == end) {
                    const auto &dest = node->output(0);
                    loop_out_outputs.push_back(dest);
                    loop_out_exprs.push_back(*expr_it);
                }
            }
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
        const auto& dst_dim = dst_layout[dim_idx];
        max_relevant_dim_size = std::max(dst_tensor[dst_dim], max_relevant_dim_size);
        if (loop_layout != expr->get_inputs()[0]->get_layout())
            throw ngraph_error("InitLoops noticed an attempt to init loop with inconsistent input layouts");
    }
    for (const auto& expr : loop_out_exprs) {
        const auto& out_tds = expr->get_outputs();
        const auto& dst_layout = out_tds[0]->get_layout();
        const auto& dst_tensor = out_tds[0]->get_tensor();
        const auto& dst_dim = dst_layout[dim_idx];
        max_relevant_dim_size = std::max(dst_tensor[dst_dim], max_relevant_dim_size);
        if (loop_layout != expr->get_outputs()[0]->get_layout())
            throw ngraph_error("InitLoops noticed an attempt to init loop with inconsistent input layouts");
    }
    for (const auto& expr : loop_in_exprs) {
        const auto& out_tds = expr->get_outputs();
        const auto& src_tensor = expr->get_inputs().front()->get_tensor();
        const auto& dst_layout = out_tds[0]->get_layout();
        const auto& dst_dim = dst_layout[dim_idx];
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
        const auto& dst_dim = loop_layout[dim_idx];
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

bool InitLoops::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;
    const auto& lowering_config = linear_ir.get_config();
    const auto loop_depth = lowering_config.m_loop_depth;
    std::stack<bool> loop_parenthesis; // 0 - begin, 1 - end

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& node = expr_it->get()->get_node();
        const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
        if (!loop_end) {
            if (ov::is_type<op::LoopBegin>(node))
                loop_parenthesis.push(0);
            continue;
        }

        const auto loop_begin = loop_end->get_loop_begin();
        auto loop_end_pos = expr_it;
        auto loop_begin_pos = std::find(linear_ir.begin(), loop_end_pos, linear_ir.get_expr_by_node(loop_begin));
        OPENVINO_ASSERT(loop_begin_pos != loop_end_pos, "LoopEnd must have corresponding LoopBegin in Linear IR!");
        const size_t dim_idx = loop_depth - loop_parenthesis.size(); // Inner = 0, Outer = 1

        OutputVector loop_in_outputs, loop_out_outputs;
        std::vector<LoweredExprPtr> loop_in_exprs, loop_out_exprs;
        get_io_exprs(linear_ir, loop_begin_pos, loop_end_pos,
                     loop_in_exprs, loop_out_exprs,
                     loop_in_outputs, loop_out_outputs);

        auto prev_expr = std::prev(loop_end_pos);
        while (ov::is_type<op::LoopEnd>((*prev_expr)->get_node())) { prev_expr = std::prev(prev_expr); }
        const auto& out_td = prev_expr->get()->get_outputs().front();
        const auto& tensor_out = out_td->get_tensor();
        const auto& subtensor_in = loop_in_exprs[0]->get_outputs().front()->get_subtensor();

        const auto& layout_out = out_td->get_layout();
        const auto dim = layout_out.size() > dim_idx + 1 ? *(layout_out.rbegin() + dim_idx) : 0;
        const auto work_amount = tensor_out.size() > dim ? tensor_out[dim] : 0;
        const auto work_amount_increment = subtensor_in.size() > dim_idx ? *(subtensor_in.rbegin() + dim_idx) :
                                           dim_idx == 0 ? m_vector_size : 1;
        const bool has_outer_loop = loop_parenthesis.size() > 1; // we have pushed the current Loop on stack, so > 1

        // If we don't need Loop (explicit execution without cycles), we can remove extra Loops
        const bool explicit_execution = work_amount == 0 || dim_idx == 0 && subtensor_in.size() > 1 && subtensor_in.back() == work_amount;
        if (explicit_execution) {
            expr_it = std::prev(loop_begin_pos);
            linear_ir.erase(loop_begin_pos);
            linear_ir.erase(loop_end_pos);
        } else {
            auto ptr_increments = init_ptr_increments(linear_ir, loop_in_exprs, loop_out_exprs, dim);
            auto finalization_offsets = init_finalization_offsets(ptr_increments, work_amount);

            OutputVector managed_outputs = loop_in_outputs;
            managed_outputs.insert(managed_outputs.end(), loop_out_outputs.begin(), loop_out_outputs.end());
            managed_outputs.push_back(loop_begin->output(0));
            loop_end->set_arguments(managed_outputs);

            // set internal flag to enable scalar vs vector loop optimizations
            loop_end->has_outer_loop = has_outer_loop;
            loop_end->set_work_amount(work_amount);
            loop_end->set_increment(work_amount_increment);
            loop_end->set_ptr_increments(ptr_increments);
            loop_end->set_finalization_offsets(finalization_offsets);

            std::vector<TensorDescriptorPtr> loop_end_inputs;
            for (const auto& expr : loop_in_exprs)
                loop_end_inputs.push_back(expr->get_inputs().front());
            for (const auto& expr : loop_out_exprs)
                loop_end_inputs.push_back(expr->get_outputs().front());
            loop_end_inputs.push_back(linear_ir.get_expr_by_node(loop_begin)->get_outputs().front());

            // TODO: set_inputs instead of new expr?
            const auto pos = std::next(loop_end_pos);
            linear_ir.erase(loop_end_pos);
            expr_it = linear_ir.insert(pos, std::make_shared<LoweredExpr>(loop_end, loop_end_inputs));
        }
        // Close parenthesis
        loop_parenthesis.pop();

        linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                            "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    }
    OPENVINO_ASSERT(loop_parenthesis.size() == 0, "All Loops must have the both LoopBegin and LoopEnd");

    return true;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

