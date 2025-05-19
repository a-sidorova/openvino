// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b_buffer_expressions.hpp"

#include <memory>

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "utils/general_utils.h"

using namespace ov::intel_cpu::brgemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov::intel_cpu {

RepackedWeightsBufferExpression::RepackedWeightsBufferExpression(
    const std::shared_ptr<ov::Node>& n,
    const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory)
    : BufferExpression(n, factory) {}

snippets::lowered::ExpressionPtr RepackedWeightsBufferExpression::clone() const {
    return std::make_shared<RepackedWeightsBufferExpression>(*this);
}

void RepackedWeightsBufferExpression::validate() const {
    BufferExpression::validate();
    OPENVINO_ASSERT(get_input_count() == 1, "RepackedWeightsBufferExpression must have only one input");
    const auto& parent_out = get_input_port_connector(0)->get_source();
    OPENVINO_ASSERT(
        ov::is_type<ov::intel_cpu::BrgemmCopyB>(parent_out.get_expr()->get_node()) && parent_out.get_index() == 0,
        "RepackedWeightsBufferExpression expects BrgemmCopyB as parent expression");
}

void RepackedWeightsBufferExpression::init_allocation_size(
    [[maybe_unused]] const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager,
    [[maybe_unused]] size_t allocation_rank) {
    const auto& parent_expr = get_input_port_connector(0)->get_source().get_expr();
    const auto& copy_b = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(parent_expr->get_node());
    const auto& brgemm_config = copy_b->get_config();

    const auto& in_subtensor = ov::snippets::utils::get_projected_subtensor(parent_expr->get_input_port(0));
    const auto n_blk = *in_subtensor.rbegin();
    const auto k_blk = *++in_subtensor.rbegin();

    const auto& precision = get_node()->get_input_element_type(0);
    const auto buffer_b_shape =
        brgemm_utils::repacking::compute_buffer_b_allocation_shape(k_blk, n_blk, brgemm_config.buffer_k_alignment(), brgemm_config.buffer_n_alignment());
    OPENVINO_ASSERT(buffer_b_shape.size() == 3, "Unexpected buffer B shape rank");
    m_allocation_size =
        std::accumulate(buffer_b_shape.cbegin(), buffer_b_shape.cend(), size_t(1), [](size_t a, size_t b) {
            return snippets::utils::dynamic_safe_mul(a, b);
        });
}

CompensationsBufferExpression::CompensationsBufferExpression(
    const std::shared_ptr<ov::Node>& n,
    const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory)
    : BufferExpression(n, factory) {}

snippets::lowered::ExpressionPtr CompensationsBufferExpression::clone() const {
    return std::make_shared<CompensationsBufferExpression>(*this);
}

void CompensationsBufferExpression::validate() const {
    BufferExpression::validate();
    OPENVINO_ASSERT(get_input_count() == 1, "CompensationsBufferExpression must have only one input");
    const auto& parent_out = get_input_port_connector(0)->get_source();
    OPENVINO_ASSERT(
        ov::is_type<ov::intel_cpu::BrgemmCopyB>(parent_out.get_expr()->get_node()) && parent_out.get_index() == 1,
        "CompensationsBufferExpression expects BrgemmCopyB as parent expression");
}

void CompensationsBufferExpression::init_allocation_size(
    [[maybe_unused]] const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager,
    [[maybe_unused]] size_t allocation_rank) {
    const auto& parent_expr = get_input_port_connector(0)->get_source().get_expr();
    const auto& copy_b = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(parent_expr->get_node());
    const auto& brgemm_config = copy_b->get_config();

    const auto n_blk = *ov::snippets::utils::get_projected_subtensor(parent_expr->get_input_port(0)).rbegin();
    m_allocation_size = compute_aligned_n_dim(n_blk, brgemm_config.buffer_n_alignment());
}

}  // namespace ov::intel_cpu
