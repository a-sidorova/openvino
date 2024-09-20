// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "need_fallback.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"


namespace ov {
namespace intel_cpu {
namespace pass {

namespace {
bool is_transposed_b(const ov::snippets::lowered::ExpressionPtr& expr) {
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::BrgemmCPU>(expr->get_node()), "Unexpected node!");
    auto layout = expr->get_input_port_descriptor(1)->get_layout();
    const auto wei_parent_expr = expr->get_input_port_connector(1)->get_source().get_expr();
    if (ov::is_type<ov::intel_cpu::BrgemmCopyB>(wei_parent_expr->get_node())) {
        layout = wei_parent_expr->get_input_port_descriptor(0)->get_layout();
    }
    return !layout.empty() && layout.back() != layout.size() - 1;
}
}  // namespace

using namespace ov::intel_cpu::brgemm_utils;

void NeedFallbackOnCPUGraph::run(const ov::snippets::lowered::LinearIR& linear_ir) {
    m_is_needed = false;
    // If the LinearIR doesn't contain domain sensitive expressions - no need to fallback
    if (linear_ir.get_config().m_enable_domain_optimization)
        return;

    // At the moment there is only one condition for fallback - on AMX platforms.
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx))
        return;

    for (const auto& expr : linear_ir) {
        if (const auto& brgemm_cpu = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            const auto& act_prc = brgemm_cpu->get_input_element_type(0);
            if (act_prc == ov::element::f32)
                continue;

            const auto& wei_shape = ov::snippets::utils::get_planar_vdims(expr->get_input_port(1));
            const auto& K = ov::snippets::utils::size_t_to_dimension(*++wei_shape.rbegin());
            const auto is_transposed = is_transposed_b(expr);
            if (K.is_dynamic() || !with_amx(get_brgemm_type(act_prc, K, is_transposed))) {
                m_is_needed = true;
                return;
            }
        }
    }
    return;
}

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov