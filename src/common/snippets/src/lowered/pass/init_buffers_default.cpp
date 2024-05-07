// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_buffers_default.hpp"

#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InitBuffersDefault::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitBuffersDefault");

    size_t reg_group = 0;
    size_t offset = 0;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto op = expr->get_node();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            AllocateBuffers::propagate_offset_to_memory_access_ops(expr, offset);
            buffer->set_reg_group(reg_group);

            offset += buffer->get_byte_size();
            reg_group++;
        }
    }

    m_buffer_scratchpad_size = offset;
    return m_buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
