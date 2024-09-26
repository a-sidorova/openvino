// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>


namespace ov {
namespace intel_cpu {

struct BrgemmCopyAKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    BrgemmCopyAKernelConfig() = default;
    BrgemmCopyAKernelConfig(const element::Type& src_dt, dnnl::impl::cpu::x64::cpu_isa_t isa);

    bool operator==(const BrgemmCopyAKernelConfig& rhs) const;
    bool operator!=(const BrgemmCopyAKernelConfig& rhs) const {return !(*this == rhs);}

    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmCopyAKernelConfig>(new BrgemmCopyAKernelConfig(*this));
    }

    bool is_empty() const;
    bool is_completed() const override;

    void update(dnnl_dim_t cur_M_blk, dnnl_dim_t K, dnnl_dim_t copy_A_wei_stride, dnnl_dim_t LDA);

    size_t hash() const override { return m_hash; }

    dnnl_data_type_t get_src_dt() const { return m_static_params->src_dt; }
    dnnl::impl::cpu::x64::cpu_isa_t get_isa() const { return m_static_params->isa; }

    dnnl_dim_t get_curr_M_blk() const { return m_curr_M_blk; }
    dnnl_dim_t get_K() const { return m_K; }
    dnnl_dim_t get_K_blk() const { return m_static_params->K_blk; }
    dnnl_dim_t get_K_tail() const { return rnd_up(get_K() % get_K_blk(), m_static_params->vnni_factor); }
    dnnl_dim_t get_copy_A_wei_stride() const { return m_copy_A_wei_stride; }
    dnnl_dim_t get_LDA() const { return m_LDA; }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

private:
    struct StaticParams {
        StaticParams(const element::Type& src_dt, dnnl::impl::cpu::x64::cpu_isa_t isa);

        const dnnl_data_type_t src_dt {dnnl_data_type_undef};
        const dnnl::impl::cpu::x64::cpu_isa_t isa {dnnl::impl::cpu::x64::isa_undef};
        const dnnl_dim_t K_blk {0};
        const size_t vnni_factor {1};
        const size_t hash {0};

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const { return !(*this == rhs); }

#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const;
#endif

    private:
        static size_t init_hash(const dnnl_data_type_t& src_dt, dnnl::impl::cpu::x64::cpu_isa_t isa, dnnl_dim_t K_blk);
    };

    size_t compute_hash() const;

    std::shared_ptr<StaticParams> m_static_params;
    dnnl_dim_t m_curr_M_blk {0};
    dnnl_dim_t m_K {0};
    dnnl_dim_t m_copy_A_wei_stride {0}, m_LDA {0};
    size_t m_hash {SIZE_MAX};
};

struct BrgemmCopyAKernel {
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t> compiled_kernel = nullptr;
};

class BrgemmCopyAKernelExecutor : public CPUKernelExecutor<BrgemmCopyAKernelConfig, BrgemmCopyAKernel> {
public:
    struct call_args {
        const void* src = nullptr;
        void* tr_src = nullptr;
    };
    BrgemmCopyAKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmCopyAKernelConfig config);

    static void execute(const BrgemmCopyAKernelExecutor* executor, call_args* args);

protected:
    std::shared_ptr<BrgemmCopyAKernel> compile_kernel(const BrgemmCopyAKernelConfig& c) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmCopyAKernelConfig& config) const override;
};
#define GET_OFF_BRGEMM_COPY_A_ARGS(field) offsetof(BrgemmCopyAKernelExecutor::call_args, field)

}   // namespace intel_cpu
}   // namespace ov
