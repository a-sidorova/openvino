// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface InsertBuffer
 * @brief
 * @param
 * @ingroup snippets
 */
class InsertBuffer : public LinearIRTransformation {
public:
    OPENVINO_RTTI("InsertBuffer", "LinearIRTransformation")
    InsertBuffer(size_t buffer_allocation_rank);
    bool run(LoweredExprIR& linear_ir) override;

private:
    void insertion(LoweredExprIR& linear_ir, const LoweredLoopManagerPtr& loop_manager, size_t loop_id,
                   const std::vector<LoweredExprPort>& loop_entries, const std::vector<LoweredExprPort>& loop_exits);

    LoweredExprIR::constExprIt insertion_position(const LoweredExprIR& linear_ir,
                                                  const LoweredLoopManagerPtr& loop_manager,
                                                  const LoweredExprPtr& up_expr, const LoweredExprPtr& down_expr);


    size_t m_buffer_allocation_rank;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
