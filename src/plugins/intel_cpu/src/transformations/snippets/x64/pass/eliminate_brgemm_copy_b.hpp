// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

#include "emitters/snippets/cpu_runtime_configurator.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface EliminateBrgemmCopyB
 * @brief EliminateBrgemmCopyB identifies BrgemmCopyB nodes which can be inferred outside the Subgraph.
 * If this is possible, CopyB node is removed, and the external repacking is configured on the further pipeline stages
 * in RuntimeConfigurator.
 *
 * @ingroup snippets
 */
class EliminateBrgemmCopyB : public ov::pass::ModelPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateBrgemmCopyB");
    EliminateBrgemmCopyB(ov::intel_cpu::RepackedInputConfigPtr repacked_inputs_config)
        : m_repacked_inputs_config(std::move(repacked_inputs_config)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::intel_cpu::RepackedInputConfigPtr m_repacked_inputs_config {nullptr};
};

}  // namespace ov::intel_cpu::pass
