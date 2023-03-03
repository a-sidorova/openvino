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
 * @interface InitLoops
 * @brief
 * @param
 * @ingroup snippets
 */
class InitLoops : public LinearIRTransformation {
public:
    OPENVINO_RTTI("InsertLoops", "LinearIRTransformation")
    InitLoops(size_t vector_size);
    bool run(LoweredExprIR& linear_ir) override;

private:
    std::vector<int64_t> init_ptr_increments(LoweredExprIR& linear_ir,
                                             const std::vector<LoweredExprPtr>& loop_in_exprs,
                                             const std::vector<LoweredExprPtr>& loop_out_exprs,
                                             size_t dim_idx) const;
    std::vector<int64_t> init_finalization_offsets(const std::vector<int64_t>& ptr_increments, size_t work_amount) const;

    size_t m_vector_size;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
