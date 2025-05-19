// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_utils.hpp"

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "utils/general_utils.h"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::snippets::utils;

namespace ov {

namespace intel_cpu::brgemm_utils {

BrgemmConfig::BrgemmConfig(const ov::element::Type& src_dt, const ov::element::Type& wei_dt, bool is_fc, bool transposed_b)  {
    const auto is_fp32 = src_dt == ov::element::f32 && wei_dt == ov::element::f32;
    const auto is_fp16 = src_dt == ov::element::f16 && wei_dt == ov::element::f16;
    const auto is_bf16 = src_dt == ov::element::bf16 && wei_dt == ov::element::bf16;
    const auto is_int8 = utils::one_of(src_dt, ov::element::i8, ov::element::u8) && wei_dt == ov::element::i8;
    OPENVINO_ASSERT(is_fp32 || is_fp16 || is_bf16 || is_int8, "Incorrect configuration");

    // Init ISA
    if (is_bf16) {
        m_isa = mayiuse(avx512_core_amx) ? avx512_core_amx :
                mayiuse(avx512_core_bf16) ? avx512_core_bf16 : isa_undef;
    } else if (is_fp16) {
        m_isa = mayiuse(avx512_core_amx_fp16) ? avx512_core_amx : isa_undef;
    } else if (is_int8) {
        m_isa = mayiuse(avx512_core_amx) ? avx512_core_amx :
                mayiuse(avx512_core_vnni) ? avx512_core_vnni :
                mayiuse(avx2_vnni_2) ? avx2_vnni_2 :
                mayiuse(avx2_vnni) ? avx2_vnni : isa_undef;
    } else if (is_fp32) {
        m_isa = mayiuse(avx512_core) ? avx512_core :
                mayiuse(cpu::x64::avx2) ? cpu::x64::avx2 : isa_undef;
    }
    OPENVINO_ASSERT(m_isa != isa_undef, "ISA is undefined!");

    // FC always requires weight repacking
    m_with_wei_repacking = !is_fp32 || transposed_b || is_fc;
    m_with_compensations = src_dt == ov::element::i8 && !one_of(m_isa, avx512_core_amx, avx2_vnni_2);
    m_with_wsp = is_amx();

    init_wei_params(is_fc, transposed_b, wei_dt);
    validate();
}

BrgemmConfig::BrgemmConfig(cpu_isa_t isa, bool is_fc, bool transposed_b, const ov::element::Type& wei_dt,
                           bool with_wei_repacking, bool with_compensations, bool with_wsp)
    : m_isa(isa), m_with_wei_repacking(with_wei_repacking), m_with_compensations(with_compensations),
    m_with_wsp(with_wsp) {
    init_wei_params(is_fc, transposed_b, wei_dt);
    validate();
}

bool BrgemmConfig::is_amx() const { 
    return is_superset(m_isa, cpu_isa_t::amx_tile);
}

void BrgemmConfig::validate() const {
    OPENVINO_ASSERT(m_isa != isa_undef, "ISA is undefined");
    OPENVINO_ASSERT(IMPLICATION(m_with_wsp, is_amx()), "Scratchpad with empty memory is withed only for AMX");
    OPENVINO_ASSERT(IMPLICATION(m_with_compensations, !is_amx() && m_with_wei_repacking), "Compensations must be only with BrgemmCopyB on non-amx platforms");
    OPENVINO_ASSERT(ov::snippets::utils::is_full_dim_value(m_wei_n_blk) || m_wei_n_blk > 0, "wei_n_blk must be positive or full dim value");
    OPENVINO_ASSERT(m_wei_k_blk > 0, "wei_k_blk must be positive");
    OPENVINO_ASSERT(m_buffer_n_alignment > 0 && m_buffer_k_alignment > 0, "buffer alignment values must be positive");
}

void BrgemmConfig::init_wei_params(bool is_fc, bool transposed_b, const ov::element::Type& wei_dt) {
    if (is_fc) {
        // TODO: Add more logic based on shapes and prc
        m_wei_n_blk = 64;
    } else {
        m_wei_n_blk = ov::snippets::utils::get_full_dim_value();
    }
    m_wei_k_blk = brgemm_utils::get_elems_in_vec(wei_dt);

    // For the details, please see 'copy_4x64' and 'copy_2x32' implementations and
    // usage in onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
    m_buffer_n_alignment = 16 * wei_dt.size();
    // For the details, please see 'transpose16x8' and 'fixup16x16' implementations and usage in
    // onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
    m_buffer_k_alignment = transposed_b ? brgemm_utils::get_elems_in_vec(wei_dt) : brgemm_utils::compute_vnni_factor(wei_dt);
}

size_t compute_vnni_factor(const ov::element::Type& precision) {
    return data_type_vnni_granularity(
        static_cast<dnnl_data_type_t>(ov::intel_cpu::DnnlExtensionUtils::ElementTypeToDataType(precision)));
}

size_t get_elems_in_vec(const ov::element::Type& precision) {
    using namespace dnnl::impl::cpu;
    OV_CPU_JIT_EMITTER_ASSERT(x64::mayiuse(x64::avx2), "doesn't support non avx512 platforms");
    const auto vlen =
        x64::mayiuse(avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen : x64::cpu_isa_traits<x64::avx2>::vlen;
    return vlen / precision.size();
}

namespace repacking {
ov::snippets::VectorDims compute_buffer_b_allocation_shape(size_t K, size_t N, size_t buffer_k_alignment, size_t buffer_n_alignment) {
    const size_t new_N = ov::snippets::utils::rnd_up(N, buffer_n_alignment);
    const size_t new_K = ov::snippets::utils::rnd_up(K, buffer_k_alignment);
    return ov::snippets::VectorDims{new_K, new_N, buffer_k_alignment};
}

ov::snippets::lowered::ExpressionPtr get_copy_b_expr(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) {
    OPENVINO_ASSERT(ov::is_type<BrgemmCPU>(brgemm_expr->get_node()),
                    "get_copy_b_expr must be called only for BrgemmCPU node");
    auto b_input_expr = brgemm_expr->get_input_port_connector(1)->get_source().get_expr();
    if (ov::is_type<BrgemmCopyB>(b_input_expr->get_node())) {
        return b_input_expr;
    }
    if (ov::is_type<snippets::lowered::BufferExpression>(b_input_expr)) {
        OPENVINO_ASSERT(b_input_expr->get_input_count() >= 1,
                        "BufferExpression on brgemm's B input must have at least one input");
        auto input_buffer_expr = b_input_expr->get_input_port_connector(0)->get_source().get_expr();
        if (ov::is_type<BrgemmCopyB>(input_buffer_expr->get_node())) {
            return input_buffer_expr;
        }
    }
    return nullptr;
}
}  // namespace repacking
}  // namespace intel_cpu::brgemm_utils

bool AttributeAdapter<ov::intel_cpu::brgemm_utils::BrgemmConfig>::visit_attributes(AttributeVisitor& visitor) {
    bool with_wei_repacking = m_ref.with_wei_repacking();
    bool with_comps = m_ref.with_compensations();
    bool with_wsp = m_ref.with_wsp();
    std::string isa = isa2str(m_ref.isa());
    size_t wei_n_blk = m_ref.wei_n_blk();
    size_t wei_k_blk = m_ref.wei_k_blk();
    size_t buffer_n_alignment = m_ref.buffer_n_alignment();
    size_t buffer_k_alignment = m_ref.buffer_k_alignment();

    visitor.on_attribute("with_brgemm_copy_b", with_wei_repacking);
    visitor.on_attribute("with_compensations", with_comps);
    visitor.on_attribute("with_wsp", with_wsp);
    visitor.on_attribute("prim_isa", isa);
    visitor.on_attribute("wei_n_blk", wei_n_blk);
    visitor.on_attribute("wei_k_blk", wei_k_blk);
    visitor.on_attribute("buffer_n_alignment", buffer_n_alignment);
    visitor.on_attribute("buffer_k_alignment", buffer_k_alignment);

    return true;
}

}  // namespace ov
