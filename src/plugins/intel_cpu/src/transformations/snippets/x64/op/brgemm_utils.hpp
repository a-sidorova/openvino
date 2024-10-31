// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/dimension.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

class BrgemmConfig {
public:
    BrgemmConfig() = default;
    BrgemmConfig(ov::element::Type src_dt, ov::element::Type wei_dt, bool transposed_b);
    BrgemmConfig(ov::element::Type src_dt, dnnl::impl::cpu::x64::cpu_isa_t isa,
                 bool need_copy_b = false, bool need_compensations = false, bool need_wsp = false);

    dnnl::impl::cpu::x64::cpu_isa_t isa() const { return m_isa; }
    bool is_amx() const { return m_isa == dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_amx; }
    bool need_copy_a(size_t K) const;
    bool need_copy_b() const { return m_need_copy_b; }
    bool need_compensations() const { return m_need_compensations; }
    bool need_wsp() const { return m_need_wsp; }

    bool need_copy_a_only_for_tail(size_t K) const;

private:
    void validate() const;

    dnnl::impl::cpu::x64::cpu_isa_t m_isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
    bool m_need_copy_b = false;
    bool m_need_compensations = false;
    bool m_need_wsp = false;

    const ov::element::Type m_src_dt = ov::element::undefined;
};

/// \brief Computes VNNI factor used by OneDNN implementation. Depends on tensor precision
size_t compute_vnni_factor(const ov::element::Type& precision);
/// \brief Computes number of elems with requested precision that fit in the vector register
size_t get_elems_in_vec(const ov::element::Type& precision);

namespace repacking {
/// \brief  Computes inner N block size used by OneDNN implementation. Depends on tensor precision
size_t compute_inner_n_block(const ov::element::Type& precision);
/// \brief  Computes inner K block size used by OneDNN implementation. Depends on tensor precision
size_t compute_inner_k_block(const ov::element::Type& precision);

/**
 * @brief Computes leading dimension (LDA) which must be used in brgemm and brgemm_copy_a emitters
 * @param k_block K block size shared between BrgemmCPU and BrgemmCopyA node
 * @param precision tensor precision
 */
template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
T compute_LDA(const T k_block, const ov::element::Type& precision) {
    return rnd_up(k_block, compute_inner_k_block(precision));
}
/**
 * @brief Computes leading dimension (LDB) which must be used in brgemm and brgemm_copy_b emitters
 * @param n_block N block size shared between BrgemmCPU and BrgemmCopyB node
 * @param precision tensor precision
 */
template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
T compute_LDB(T n_block, const ov::element::Type& precision) {
    return snippets::utils::is_dynamic_value<T>(n_block) ?
           n_block :
           std::max(n_block, static_cast<T>(compute_inner_n_block(precision)));
}
}   // namespace repacking
}   // namespace brgemm_utils
}   // namespace intel_cpu
template <>
class AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig> : public VisitorAdapter {
public:
    AttributeAdapter(intel_cpu::brgemm_utils::BrgemmConfig& ref) : m_ref(ref) {}
    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig>");

protected:
    intel_cpu::brgemm_utils::BrgemmConfig& m_ref;
};
}   // namespace ov
