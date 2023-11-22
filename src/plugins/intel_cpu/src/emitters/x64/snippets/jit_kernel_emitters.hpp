// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_container_emitter.hpp"

#include "jit_snippets_dynamic_emitter.hpp"
#include "snippets/op/loop.hpp"

namespace ov {
namespace intel_cpu {

///
/// \brief    Kernel is the only entry point to Codogen Jit compilation. Kernel perform abstract-to-physical register
/// mapping and creates a pools of available gpr and vec registers. Kernel usually contains (at least one)
/// LoopBeginEmitter and LoopEndEmitter pair. In general the enclosed emitters should be organized in the following way:
/// jit_kernel_emitter {                 /* entry point, maps registers, creates pools of available registers */
///     1.S LoopBeginEmitter        /* Scalar Loop over the outer dimension [START] */
///         2.S LoopBeginEmitter    /* inner vector loop [START] */
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         2.E LoopEndEmitter      /* inner vector loop [END] */
///         3.S LoopBeginEmitter    /* inner scalar loop for tail processing [START]*/
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         3.E LoopEndEmitter      /* inner scalar loop for tail processing [END]*/
///     1.E LoopEndEmitter          /* Scalar Loop over the outer dimension [END] */
/// }
/// Note that Kernel doesn't accept any input arguments.
///

class jit_kernel_emitter : public jit_container_emitter {
public:
    jit_kernel_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    /**
    * @brief populates physical registers pools for x86 (both vec and gp).
     * Skips stack-related gprs and extra gprs passed as arguments.
     * @arg gpr_blacklist - set of gp registers that should not be added to register pool
     * @arg vec_blacklist - set of vec registers should not be added to register pool
    */
    void init_reg_pools(const std::set<size_t>& gpr_blacklist, const std::set<size_t>& vec_blacklist);

    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    std::vector<size_t> vec_regs_pool;
    std::vector<size_t> gp_regs_pool;
    size_t num_inputs{0};
    size_t num_outputs{0};
    size_t num_unique_buffers{0};
    snippets::lowered::LinearIR::container mem_access_exprs;
    snippets::lowered::LinearIR::container general_exprs;
};

class jit_kernel_static_emitter : public jit_kernel_emitter {
public:
    jit_kernel_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr);

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void init_data_pointers(const Xbyak::Reg64& reg_indexes, const Xbyak::Reg64& reg_const_params, const std::vector<Xbyak::Reg64>& data_ptr_regs) const;

    std::vector<std::vector<int64_t>> data_offsets;
    const size_t reg_indexes_idx{0};
    const size_t reg_runtime_params_idx{0};
    std::vector<size_t> master_shape;
};

class jit_kernel_dynamic_emitter : public jit_kernel_emitter, public jit_snippets_dynamic_emitter {
public:
    jit_kernel_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr);

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void init_data_pointers(const Xbyak::Reg64& reg_runtime_params, const std::vector<Xbyak::Reg64>& data_ptr_regs) const;

    const size_t reg_runtime_params_idx{0};
};

}   // namespace intel_cpu
}   // namespace ov
