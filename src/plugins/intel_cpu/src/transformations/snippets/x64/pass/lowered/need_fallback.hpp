// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface NeedFallbackOnCPUGraph
 * @brief Check if the current LinearIR is not supported and there should be fallback on CPU Graph
 *        Current conditions for fallback:
 *          - There is BrgemmCPU INT8|BF16 with K % VNNI_FACTOR != 0 in AMX case.
 * @ingroup snippets
 */
class NeedFallbackOnCPUGraph : public ov::snippets::lowered::pass::AnalyzerPass {
public:
    OPENVINO_RTTI("NeedFallbackOnCPUGraph", "BrgemmBlocking")
    NeedFallbackOnCPUGraph(bool& is_needed) : m_is_needed(is_needed) {}

    void run(const ov::snippets::lowered::LinearIR& linear_ir) override;

private:
    bool& m_is_needed;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov