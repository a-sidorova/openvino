// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {

namespace intel_cpu::brgemm_utils {

class BrgemmConfig {
public:
    BrgemmConfig() = default;
    BrgemmConfig(const ov::element::Type& src_dt, const ov::element::Type& wei_dt, bool is_fc, bool transposed_b);
    BrgemmConfig(dnnl::impl::cpu::x64::cpu_isa_t isa, bool is_fc, bool transposed_b, const ov::element::Type& wei_dt,
                 bool with_wei_repacking = false, bool with_compensations = false, bool with_wsp = false);

    dnnl::impl::cpu::x64::cpu_isa_t isa() const { return m_isa; }
    bool is_amx() const;
    bool with_wei_repacking() const { return m_with_wei_repacking; }
    bool with_compensations() const { return m_with_compensations; }
    bool with_wsp() const { return m_with_wsp; }
    bool with_scratchpad() const { return with_wsp() || with_compensations(); }
    size_t wei_n_blk() const { return m_wei_n_blk; }
    size_t wei_k_blk() const { return m_wei_k_blk; }
    size_t buffer_n_alignment() const { return m_buffer_n_alignment; }
    size_t buffer_k_alignment() const { return m_buffer_k_alignment; }

private:
    void validate() const;
    void init_wei_params(bool is_fc, bool transposed_b, const ov::element::Type& wei_dt);

    dnnl::impl::cpu::x64::cpu_isa_t m_isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
    bool m_with_wei_repacking = false;
    bool m_with_compensations = false;
    bool m_with_wsp = false;
    size_t m_wei_n_blk = 0lu;
    size_t m_wei_k_blk = 0lu;
    size_t m_buffer_n_alignment = 0lu;
    size_t m_buffer_k_alignment = 0lu;
};

/// \brief Computes VNNI factor used by OneDNN implementation. Depends on tensor precision
size_t compute_vnni_factor(const ov::element::Type& precision);
/// \brief Computes number of elems with requested precision that fit in the vector register
size_t get_elems_in_vec(const ov::element::Type& precision);

namespace repacking {
/// \brief  Computes N dim in output blocked shape of BrgemmCopyB. Depends on tensor precision
template <typename T,
          typename = typename std::enable_if_t<(std::is_same_v<T, size_t> || std::is_same_v<T, int64_t>), bool>>
inline T compute_aligned_n_dim(T n, size_t buffer_n_alignment) {
    return ov::snippets::utils::rnd_up(n, static_cast<T>(buffer_n_alignment));
}

/// \brief  Computes N dim in output blocked shape of BrgemmCopyB. Depends on tensor precision
inline size_t compute_LDB(size_t n, size_t wei_n_blk, size_t buffer_n_alignment) {
    return ov::snippets::utils::is_full_dim_value(wei_n_blk) ?
                ov::snippets::utils::rnd_up(n, buffer_n_alignment) :
                wei_n_blk;
}

/// \brief  Computes allocation shape for Buffer between BrgemmCopyB and Brgemm
ov::snippets::VectorDims compute_buffer_b_allocation_shape(size_t K, size_t N, size_t buffer_k_alignment, size_t buffer_n_alignment);

/**
 * @brief Retrieves the expression pointer for the brgemm_copy_b expression corresponding to the given BrgemmCPU
 * expression.
 * @param brgemm_expr The expression pointer for the BrgemmCPU operation.
 * @return The expression pointer for the BrgemmCopyB operation.
 */
snippets::lowered::ExpressionPtr get_copy_b_expr(const snippets::lowered::ExpressionPtr& brgemm_expr);
}  // namespace repacking
}  // namespace intel_cpu::brgemm_utils

template <>
class AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig> : public VisitorAdapter {
public:
    AttributeAdapter(intel_cpu::brgemm_utils::BrgemmConfig& ref) : m_ref(ref) {}
    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig>");

protected:
    intel_cpu::brgemm_utils::BrgemmConfig& m_ref;
};

}  // namespace ov
