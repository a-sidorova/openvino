// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface LoadMoveBroadcastToBroadcastLoad
 * @brief Fuses consecutive Load and MoveBroadcast into a single load insctruction.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class LoadMoveBroadcastToBroadcastLoad: public LinearIRTransformation {
public:
    LoadMoveBroadcastToBroadcastLoad();
    OPENVINO_RTTI("SetScalarCountForLoadStore", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

}  // namespace lowered
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
