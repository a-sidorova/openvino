// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitters.hpp"

#include "jit_snippets_dynamic_emitter.hpp"

#include <cpu/x64/jit_generator.hpp>


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;


namespace ov {
namespace intel_cpu {

using namespace dnnl::impl::cpu::x64;

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<Xbyak::Reg64>& regs) {
    regs.resize(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){ return Xbyak::Reg64(static_cast<int>(idx)); });
}

jit_kernel_emitter::jit_kernel_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_container_emitter(h, isa) {
    const auto kernel = ov::as_type_ptr<snippets::op::Kernel>(expr->get_node());
    OPENVINO_ASSERT(kernel, "jit_kernel_emitter invoked with invalid op argument");
    OPENVINO_ASSERT(!kernel->region.empty(), "jit_kernel_emitter invoked with empty body");
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    num_inputs = 0;
    num_outputs = 0;
    const auto& io_exprs = body.get_IO_ops();
    for (const auto& expr : io_exprs) {
        mem_access_exprs.push_back(expr);
        if (expr->get_type() == snippets::lowered::IOExpression::io_type::INPUT) {
            num_inputs++;
            continue;
        }
        if (expr->get_type() == snippets::lowered::IOExpression::io_type::OUTPUT) {
            num_outputs++;
            continue;
        }
        OPENVINO_THROW("jit_kernel_emitter detected unsupported io_type");
    }
    std::set<size_t> unique_buffers;
    for (const auto& expr : body) {
        if (const auto buffer = ov::as_type_ptr<snippets::op::Buffer>(expr->get_node())) {
            const auto buffer_id = buffer->get_id();
            if (unique_buffers.count(buffer_id) == 0) {
                mem_access_exprs.push_back(expr);
                unique_buffers.insert(buffer_id);
            }
        } else {
            general_exprs.emplace_back(expr);
        }
    }
    num_unique_buffers = unique_buffers.size();
}

void jit_kernel_emitter::init_reg_pools(const std::set<size_t>& gpr_blacklist, const std::set<size_t>& vec_blacklist) {
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 16; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 15 - i;
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                  [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    std::set<size_t> gprs_blacklist_extended{Xbyak::Operand::RSP, Xbyak::Operand::RBP};
    gprs_blacklist_extended.insert(gpr_blacklist.begin(), gpr_blacklist.end());
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, gprs_blacklist_extended);
    remove_regs_from_pool(vec_regs_pool, vec_blacklist);
}

void jit_kernel_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_kernel_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OPENVINO_ASSERT(in.empty() && out.empty(), "jit_kernel_emitter expects 0 registers on input and output");
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    OPENVINO_ASSERT(data_ptr_regs_idx.size() == num_params,
                    "jit_kernel_emitter: number of inputs and outputs is inconsistent with the number of allocated registers ", num_params,
                    " data_ptr_regs_idx.size() = ", data_ptr_regs_idx.size());
}

jit_kernel_static_emitter::jit_kernel_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr), reg_indexes_idx(abi_param1.getIdx()), reg_runtime_params_idx(abi_param2.getIdx()) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OPENVINO_ASSERT(kernel, "jit_kernel_static_emitter expectes KernelStatic expression");
    data_offsets = kernel->get_data_offsets();
    master_shape = body.get_master_shape();
    // Note: plugin can prepend master shape with 1 to facilitate parallel execution (usually up to 6D tensor)
    //       so we have to reproduce this behavior here
    master_shape.insert(master_shape.begin(), jcp.parallel_executor_ndims - master_shape.size(), 1);

    // Initialize pools of gp and vec registers
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    init_reg_pools({reg_indexes_idx, reg_runtime_params_idx}, {});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);

    // Note that we can't use reg_indexes_idx or reg_runtime_params_idx to store data pointers because these two
    // regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);
    // However we can use reg_indexes_idx and reg_runtime_params_idx for other operations since we won't need them
    // after offsets calculation
    gpr_map_pool.second.push_back(reg_indexes_idx);
    gpr_map_pool.second.push_back(reg_runtime_params_idx);
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}


void jit_kernel_static_emitter::init_data_pointers(const Xbyak::Reg64& reg_indexes, const Xbyak::Reg64& reg_const_params,
                                                   const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    const auto num_params = num_inputs + num_outputs;
    const auto offset_rank = master_shape.size() - 1;
    OPENVINO_ASSERT(!data_offsets.empty() && data_offsets.front().size() == master_shape.size(), "Incorrect data_offsets value!");

    std::function<void(Xbyak::Reg64, const std::vector<int64_t>&, Xbyak::Reg64)> init_ptr_with_offset;
    init_ptr_with_offset = [&](Xbyak::Reg64 pointer, const std::vector<int64_t>& offsets, Xbyak::Reg64 reg_tmp) {
        for (size_t j = 0; j < offset_rank; j++) {
            if (master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(int64_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    const auto spare_corruptable_gpr = std::find_if(gp_regs_pool.begin(), gp_regs_pool.end(),
                                                   [this](size_t reg) {
                                                        return reg != reg_indexes_idx && reg != reg_runtime_params_idx;
                                                   });
    const bool last_iter_explicitly = spare_corruptable_gpr == gp_regs_pool.end();
    Xbyak::Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs[num_params - 1] : Xbyak::Reg64(static_cast<int>(*spare_corruptable_gpr));
    // Vector "data_ptr_regs" is sorted by abstract regs.
    // It means that the vector contains the physical registers in order [src, .., src, dst, .., dst, buffer]
    // So we can initialize buffer register firstly as last value of vector "data_ptr_regs"
    // NOTE: Snippets Buffer Scratchpad has the common data pointer for all Buffers (even with different ID).
    //       The accessing memory is covered by correct offsets in each Buffer and the corresponding MemoryAccess ops
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_const_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_const_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_const_params to pass runtime args to LoopScheduler, so we have to
    //     push a reg on the stack, and restore it value afterwards
    if (last_iter_explicitly) {
        h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_const_params;
        // can corrupt reg_const_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
}

void jit_kernel_static_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    h->preamble();

    Xbyak::Reg64 reg_indexes = Xbyak::Reg64(static_cast<int>(reg_indexes_idx));
    Xbyak::Reg64 reg_const_params = Xbyak::Reg64(static_cast<int>(reg_runtime_params_idx));
    std::vector<Xbyak::Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_regs_idx, data_ptr_regs);

    init_data_pointers(reg_indexes, reg_const_params, data_ptr_regs);
    for (const auto& expression : body) {
        const auto& emitter = expression->get_emitter();
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = expression->get_reg_info();
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }
    h->postamble();
}

jit_kernel_dynamic_emitter::jit_kernel_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr), reg_runtime_params_idx(abi_param1.getIdx()) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelDynamic>(expr->get_node());
    OPENVINO_ASSERT(kernel, "jit_kernel_dynamic_emitter expectes KernelDynamic expression");

    init_reg_pools({reg_runtime_params_idx}, {});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}

void jit_kernel_dynamic_emitter::init_data_pointers(const Xbyak::Reg64& reg_runtime_params, const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    const auto num_params = num_inputs + num_outputs;
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_runtime_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    for (size_t i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
    }
}

void jit_kernel_dynamic_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    h->preamble();

    Xbyak::Reg64 reg_runtime_params = Xbyak::Reg64(static_cast<int>(reg_runtime_params_idx));
    std::vector<Xbyak::Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_regs_idx, data_ptr_regs);

    init_data_pointers(reg_runtime_params, data_ptr_regs);

    for (const auto& expression : body) {
        const auto& emitter = expression->get_emitter();
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = expression->get_reg_info();
        // Note all DynamicEmitters should have access to the runtime_params argument,
        // since parameters computed by configurator are stored there.
        if (std::dynamic_pointer_cast<jit_snippets_dynamic_emitter>(emitter))
            in_regs.push_back(reg_runtime_params_idx);
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }
    h->postamble();
}

}   // namespace intel_cpu
}   // namespace ov
