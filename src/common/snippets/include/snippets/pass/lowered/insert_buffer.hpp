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
    void insertion(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos);

    size_t m_buffer_allocation_rank;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
