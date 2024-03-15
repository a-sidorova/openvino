// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface ComputeBufferAllocationSize
 * @brief The pass calculate allocation sizes of Buffers.
 * @param m_buffer_allocation_rank - rank of shape for memory allocation: shape[shape_rank - normalize(m_allocation_rank) : shape_rank]
 * @ingroup snippets
 */
class ComputeBufferAllocationSize : public RangedPass {
public:
    OPENVINO_RTTI("ComputeBufferAllocationSize", "RangedPass")
    ComputeBufferAllocationSize(size_t buffer_allocation_rank) : m_buffer_allocation_rank(buffer_allocation_rank) {}
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    static size_t get_allocation_size(const LinearIR::LoopManagerPtr& loop_manager, const ExpressionPtr& buffer_expr, size_t allocation_rank);

private:
    size_t m_buffer_allocation_rank = 0;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
