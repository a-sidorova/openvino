// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/snippets_isa.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface AllocateBuffers
 * @brief The pass calculation common size of buffer scratchpad and propagates Buffer offsets to connected MemoryAccess operations.
 * @ingroup snippets
 */

class AllocateBuffers : public Pass {
public:
    OPENVINO_RTTI("AllocateBuffers", "Pass")
    bool run(lowered::LinearIR& linear_ir) override;

    size_t get_scratchpad_size() const { return m_buffer_scratchpad_size; }

private:
    static void propagate_offset(const LinearIR& linear_ir, const ExpressionPtr& buffer_expr, size_t offset);

    size_t m_buffer_scratchpad_size = 0;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
