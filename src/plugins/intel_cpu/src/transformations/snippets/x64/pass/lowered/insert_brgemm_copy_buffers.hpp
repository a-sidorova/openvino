// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface InsertBrgemmCopyBuffers
 * @brief Insert special Buffers after BrgemmCopyA and BrgemmCopyB with algorithm of allocation size calculation which
 *        distinguishes with common algorithm
 * @ingroup snippets
 */
class InsertBrgemmCopyBuffers: public snippets::lowered::pass::RangedPass {
public:
    InsertBrgemmCopyBuffers() = default;
    OPENVINO_RTTI("InsertBrgemmCopyBuffers", "Pass");
    bool run(snippets::lowered::LinearIR& linear_ir, snippets::lowered::LinearIR::constExprIt begin, snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
