// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "external_repacking_adjuster.hpp"

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov {
namespace intel_cpu {

BrgemmExternalRepackingAdjuster::BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                                 const CPURuntimeConfigurator* configurator)
    : snippets::lowered::pass::RuntimeOptimizer(configurator) {
    const auto& params = linear_ir->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto consumers = param->get_output_port_connector(0)->get_consumers();
        const bool brgemm_with_extracted_repacking =
            std::any_of(consumers.begin(), consumers.end(), [](const ov::snippets::lowered::ExpressionPort& port) {
                auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(port.get_expr()->get_node());
                return brgemm && brgemm_utils::with_repacking(brgemm->get_type()) && port.get_index() == 1;
            });
        if (brgemm_with_extracted_repacking) {
            m_param_idces_with_external_repacking.insert(i);
            // Ticket 157339: Support non-planar layout
            OPENVINO_ASSERT(ov::snippets::utils::is_planar_layout(configurator->get_io_descs()[i]->get_layout()),
                            "Non-planar layout is not supported for external repacking");
        }
    }
}

bool BrgemmExternalRepackingAdjuster::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmExternalRepackingAdjuster")
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());
    const float L2_cache_size = dnnl::utils::get_cache_size(2, true);

    bool fit_into_L2 = true;
    for (const auto& i : m_param_idces_with_external_repacking) {
        const auto& shape = cpu_config->io_shapes[i];
        if (shape == cpu_config->latest_shapes[i])
            continue;

        const auto& K = *++shape.rbegin();
        const auto& N = *shape.rbegin();

        const auto& precision = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        const auto vnni_factor = brgemm_utils::compute_vnni_factor(precision);
        const size_t brgemm_kernel_rank = 2;
        // Firstly, batch dims are set
        VectorDims requested_blocked_shape(shape.begin(), shape.end() - brgemm_kernel_rank);
        // Then, the blocked dims are formed
        const auto new_K = snippets::utils::div_up(K, vnni_factor);
        const auto new_N = std::max(N, brgemm_utils::repacking::compute_inner_n_block(precision));
        requested_blocked_shape.insert(requested_blocked_shape.end(), {new_K, new_N, vnni_factor});

        VectorDims requested_order(shape.size() - brgemm_kernel_rank);
        std::iota(requested_order.begin(), requested_order.end(), 0);
        const auto last_idx = shape.size() - 1;
        requested_order.insert(requested_order.end(), {last_idx - 1, last_idx, last_idx - 1});

        const auto desc =
            std::make_shared<CpuBlockedMemoryDesc>(precision, Shape(shape), requested_blocked_shape, requested_order);

        auto config = BrgemmCopyBKernelConfig(precision,
                                              precision,
                                              dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_amx,
                                              false,
                                              false,
                                              brgemm_utils::repacking::compute_inner_n_block(precision));
        const auto executor = std::make_shared<BrgemmCopyBKernelExecutor>(
            static_cast<const CPURuntimeConfigurator*>(m_configurator)->get_cache(),
            config);
        config.update(N,
                      N,
                      K,
                      K,
                      ov::snippets::utils::get_dim_in_stride(shape, cpu_config->io_layouts[i], 1) * precision.size(),
                      brgemm_utils::repacking::compute_LDB(N, precision));
        executor->update_by_config(config);

        // Save original input offsets for input before repacking.
        const auto in_offsets = cpu_config->io_data_offsets[i];

        ov::snippets::VectorDims shape_for_offset(cpu_config->tensor_rank - shape.size(), 1);
        shape_for_offset.insert(shape_for_offset.end(), requested_blocked_shape.begin(), requested_blocked_shape.end());
        m_configurator->compute_offsets(shape_for_offset, i, 0);
        // Save new input offsets for input after repacking.
        const auto out_offsets = cpu_config->io_data_offsets[i];

        cpu_config->repacked_inputs[i] = CPURuntimeConfig::RepackedInput(desc, executor, in_offsets, out_offsets);

        const auto src_size = N * K * precision.size();
        const auto dst_size = new_N * new_K * precision.size();
        fit_into_L2 &= ((src_size + dst_size) < L2_cache_size);
    }

    if (!cpu_config->repacked_inputs.empty()) {
        // Heuristic: If external repacking data doesn't fit in the cache L2,
        //            external repacking should be executed in seperate parallel section before kernel execution.
        cpu_config->repacking_impl_type = fit_into_L2 ? CPURuntimeConfig::RepackingImplType::IN_PARALLEL
                                                      : CPURuntimeConfig::RepackingImplType::SEPARATE;

        // In parallel case Kernel should not add offsets to repacked inputs because
        // they will be applied during repacking in execution stage
        if (cpu_config->repacking_impl_type == CPURuntimeConfig::RepackingImplType::IN_PARALLEL) {
            for (const auto& in : cpu_config->repacked_inputs) {
                auto& offsets = cpu_config->io_data_offsets[in.first];
                std::fill(offsets.begin(), offsets.end(), 0);
            }
        }
    }

    return true;
}

}  // namespace intel_cpu
}  // namespace ov
