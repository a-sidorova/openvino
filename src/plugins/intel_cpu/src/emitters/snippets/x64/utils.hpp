// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "snippets/lowered/expression_port.hpp"

namespace ov {
namespace intel_cpu {
namespace utils {

inline static std::vector<Xbyak::Reg64> transform_idxs_to_regs(const std::vector<size_t>& idxs) {
    std::vector<Xbyak::Reg64> regs(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Xbyak::Reg64(static_cast<int>(idx));});
    return regs;
}

inline static std::vector<size_t> transform_snippets_regs_to_idxs(const std::vector<snippets::Reg>& regs) {
    std::vector<size_t> idxs(regs.size());
    std::transform(regs.cbegin(), regs.cend(), idxs.begin(), [](const snippets::Reg& reg) { return reg.idx; });
    return idxs;
}

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port, size_t offset);

Xbyak::Reg64 get_aux_gpr(const std::vector<size_t>& used_gpr_idxs);

}   // namespace utils
}   // namespace intel_cpu
}   // namespace ov