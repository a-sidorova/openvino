// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_utils.hpp"

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "utils/general_utils.h"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

BrgemmConfig::BrgemmConfig(ov::element::Type src_dt, ov::element::Type wei_dt, bool transposed_b)
    : m_src_dt(std::move(src_dt)) {
    const auto is_fp32 = m_src_dt == ov::element::f32 && wei_dt == ov::element::f32;
    const auto is_bf16 = m_src_dt == ov::element::bf16 && wei_dt == ov::element::bf16;
    const auto is_int8 = (m_src_dt == ov::element::i8 || m_src_dt == ov::element::u8) && wei_dt == ov::element::i8;
    OPENVINO_ASSERT(is_fp32 || is_bf16 || is_int8, "Incorrect configuration");

    // Init ISA
    if (is_bf16) {
        m_isa = mayiuse(avx512_core_amx) ? avx512_core_amx :
                mayiuse(avx512_core_bf16) ? avx512_core_bf16 : isa_undef;
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

    m_need_copy_b = !is_fp32 || transposed_b;

    m_need_compensations = m_src_dt == ov::element::i8 && !one_of(m_isa, avx512_core_amx, avx2_vnni_2);
    m_need_wsp = m_isa == avx512_core_amx;

    validate();
}

BrgemmConfig::BrgemmConfig(ov::element::Type src_dt, cpu_isa_t isa, bool need_copy_b, bool need_compensations, bool need_wsp)
    : m_isa(isa), m_need_copy_b(need_copy_b), m_need_compensations(need_compensations), m_need_wsp(need_wsp), m_src_dt(std::move(src_dt)) {
    validate();
}

bool BrgemmConfig::need_copy_a(size_t K) const {
    return is_amx() && (is_dynamic_value(K) || (K % compute_vnni_factor(m_src_dt) != 0));
}

bool BrgemmConfig::need_copy_a_only_for_tail(size_t K) const {
    // currently we support only AMX scenario with CopyA where we should process only [M_blk, K_Tail] tensor
    return need_copy_a(K);
}

void BrgemmConfig::validate() const {
    OPENVINO_ASSERT(m_isa != isa_undef, "ISA is undefined");
    OPENVINO_ASSERT(IMPLICATION(m_need_wsp, is_amx()), "Scratchpad with empty memory is needed only for AMX");
    OPENVINO_ASSERT(IMPLICATION(m_need_compensations, !is_amx() && m_need_copy_b), "Compensations must be only with BrgemmCopyB on non-amx platforms");
}

size_t compute_vnni_factor(const ov::element::Type& precision) {
    return data_type_vnni_granularity(static_cast<dnnl_data_type_t>(ov::intel_cpu::DnnlExtensionUtils::ElementTypeToDataType(precision)));
}

size_t get_elems_in_vec(const ov::element::Type& precision) {
    using namespace dnnl::impl::cpu;
    OV_CPU_JIT_EMITTER_ASSERT(x64::mayiuse(x64::avx2), "doesn't support non avx512 platforms");
    const auto vlen = x64::mayiuse(avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen : x64::cpu_isa_traits<x64::avx2>::vlen;
    return vlen / precision.size();
}

namespace repacking {
size_t compute_inner_n_block(const ov::element::Type& precision) {
    switch (precision) {
        case element::i8: return 64;
        case element::bf16: return 32;
        case element::f32: return 16;
        default: OPENVINO_THROW("BrgemmCopyB doesn't support precision ", precision);
    }
}

size_t compute_inner_k_block(const ov::element::Type& precision) {
    return brgemm_utils::get_elems_in_vec(precision);
}
}   // namespace repacking
}   // namespace brgemm_utils
}   // namespace intel_cpu
bool AttributeAdapter<ov::intel_cpu::brgemm_utils::BrgemmConfig>::visit_attributes(AttributeVisitor& visitor) {
    bool need_copy_b = m_ref.need_copy_b();
    bool need_comps = m_ref.need_compensations();
    bool need_wsp = m_ref.need_wsp();
    std::string isa = isa2str(m_ref.isa());

    visitor.on_attribute("with_brgemm_copy_b", need_copy_b);
    visitor.on_attribute("with_compensations", need_comps);
    visitor.on_attribute("with_wsp", need_wsp);
    visitor.on_attribute("prim_isa", isa);

    return true;
}
}   // namespace ov
