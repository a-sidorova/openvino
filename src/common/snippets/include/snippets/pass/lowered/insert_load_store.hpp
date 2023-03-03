// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface InsertLoadStore
 * @brief InsertLoadStore
 * @ingroup snippets
 */
class InsertLoadStore : public LinearIRTransformation {
public:
    explicit InsertLoadStore(size_t vector_size);
    OPENVINO_RTTI("InsertLoadStore", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;

private:
    bool insert_load(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt in_expr_it);
    bool insert_store(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt out_expr_it);

    size_t m_vector_size;
};

} //namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
