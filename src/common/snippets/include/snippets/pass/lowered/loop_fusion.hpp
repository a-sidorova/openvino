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
 * @interface LoopFusion
 * @brief
 * @param
 * @param
 * @ingroup snippets
 */
class LoopFusion : public LinearIRTransformation {
public:
    OPENVINO_RTTI("LoopFusion", "LinearIRTransformation")
    LoopFusion();
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
