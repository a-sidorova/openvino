// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

#include "emitters/snippets/cpu_runtime_configurator.hpp"

namespace ov::intel_cpu {

/**
 * @class InitRepackedConstantInputs
 * @brief TODO
 */
class InitRepackedConstantInputs : public ov::snippets::lowered::pass::ConstPass {
public:
    OPENVINO_RTTI("InitRepackedConstantInputs", "", ov::snippets::lowered::pass::ConstPass)
    InitRepackedConstantInputs() = default;
    InitRepackedConstantInputs(std::set<size_t> constant_input_idxs,
                               ov::intel_cpu::MultiCacheWeakPtr cache,
                               ov::intel_cpu::RepackedInputConfig& repacked_runtime_inputs_config,
                               ov::intel_cpu::RepackedInputConfig& repacked_const_inputs_config)
        : m_constant_input_idxs(std::move(constant_input_idxs)),
          m_cache(std::move(cache)),
          m_repacked_runtime_inputs_config(repacked_runtime_inputs_config),
          m_repacked_const_inputs_config(repacked_const_inputs_config) {}

    bool run(const snippets::lowered::LinearIR& linear_ir) override;

private:
    std::set<size_t> m_constant_input_idxs {};
    ov::intel_cpu::MultiCacheWeakPtr m_cache;
    ov::intel_cpu::RepackedInputConfig& m_repacked_runtime_inputs_config;
    ov::intel_cpu::RepackedInputConfig& m_repacked_const_inputs_config;
};

}  // namespace ov::intel_cpu
