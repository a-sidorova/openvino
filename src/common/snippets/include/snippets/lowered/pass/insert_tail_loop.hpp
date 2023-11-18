// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/op/loop.hpp"
#include "snippets/runtime_config.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertTailLoop
 * @brief Injects tail-processing loop after a vector loop if required.
 *  Additional optimizations are performed if a loop body is executed only once.
 * @ingroup snippets
 */
class InsertTailLoop : public Pass {
public:
    OPENVINO_RTTI("InsertTailLoop", "Pass")
    InsertTailLoop(ov::snippets::RuntimeConfig config) : m_runtime_config(config) {}
    bool run(LinearIR& linear_ir) override;

private:
    std::shared_ptr<op::LoopEnd> create_tail_loop(LinearIR& linear_ir,
                                                  LinearIR::constExprIt vector_begin, LinearIR::constExprIt vector_end,
                                                  LinearIR::constExprIt& tail_begin, LinearIR::constExprIt& tail_end,
                                                  const std::shared_ptr<op::LoopEnd>& vector_loop_end,
                                                  bool is_vector_inserted,
                                                  const RuntimeConfig::LoopDescriptor& tail_loop_desc,
                                                  const std::map<size_t, size_t>& updated_loop_ids);

    static void tail_transformations(LinearIR& linear_ir,
                                     LinearIR::constExprIt tail_begin,
                                     LinearIR::constExprIt tail_end,
                                     size_t tail_size);

    ov::snippets::RuntimeConfig m_runtime_config;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
