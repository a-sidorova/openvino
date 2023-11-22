// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/port_connector.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

size_t jit_brgemm_emitter::getBrgIdx(size_t kIdx, size_t nIdx) {
    return kIdx * BRGEMM_N_KERNEL_NUM + nIdx;
}

size_t jit_brgemm_emitter::get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Input shape is original, so we need to correctly read this data by order
    // Example:
    //      Original shape (shape) = [1, 49, 2, 23]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Transposed shape = [2, 1, 49, 23]
    //      The leading dimension is equal to stride of shape[layout[3]] = 2 x 23
    OPENVINO_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                    "jit_brgemm_emitter detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout[layout.size() - 2];  // `1` in example
    return std::accumulate(shape.cbegin() + idx + 1, shape.end(), 1, std::multiplies<size_t>());
}
size_t jit_brgemm_emitter::get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Output shape is already transposed, we need to correctly write the data with original shape by the order
    // Example:
    //      Original transposed shape (shape) = [49, 2, 7, 39]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Before leading dimension with index 3 there is dimension with index 2 in planar layout.
    //      Since we have non-planar layout, we have to find this before LD dim in transposed order.
    //      In layout 2nd idx is first element, it means, that the leading dimension is equal to stride of shape[0]
    OPENVINO_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                    "jit_brgemm_emitter detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout.size() - 2; // 2 in the example
    const auto dim = std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), idx)); // 0 in the example: shape[0] = 49
    return std::accumulate(shape.cbegin() + dim + 1, shape.cend(), 1, std::multiplies<size_t>()); // shape[1] x shape[2] x shape[3] = 2 x 7 x 39
}

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr) : jit_emitter(h, isa) {
    m_brgCtxs.fill(brgemmCtx());
    std::generate(m_brgKernels.begin(), m_brgKernels.end(), [](){ return nullptr; });
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    if (brgemm_node->is_dynamic())
        OPENVINO_THROW("Snippets don't support code generation for dynamic Brgemm");
    const auto brgemm_copy = brgemm_node->is_with_data_repacking() ? brgemm_node->get_brgemm_copy() : nullptr;

    std::vector<size_t> leading_dimensions;
    std::vector<std::vector<size_t>> io_layouts;

     auto get_layout = [](const std::vector<size_t>& layout, const snippets::VectorDims& io_shape) {
        if (!layout.empty()) return layout;
        std::vector<size_t> default_layout(io_shape.size());
        std::iota(default_layout.begin(), default_layout.end(), 0);
        return default_layout;
    };

    auto init_in_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& input) {
        io_layouts.push_back(get_layout(input->get_layout(), input->get_shape()));
        leading_dimensions.push_back(get_in_leading_dim(input->get_shape(), io_layouts.back()));
    };
    auto init_out_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& output) {
        io_layouts.push_back(get_layout(output->get_layout(), output->get_shape()));
        leading_dimensions.push_back(get_out_leading_dim(output->get_shape(), io_layouts.back()));
    };
    init_in_scheduling_params(expr->get_input_port_descriptor(0));
    if (brgemm_node->is_with_data_repacking()) {
        io_layouts.push_back(std::vector<size_t>{});
        leading_dimensions.push_back(0);
    } else {
        init_in_scheduling_params(expr->get_input_port_descriptor(1));
    }
    init_out_scheduling_params(expr->get_output_port_descriptor(0));

    const auto& A_shape = expr->get_input_port_descriptor(0)->get_shape();
    const auto& A_layout = io_layouts[0];
    const auto& C_shape = expr->get_output_port_descriptor(0)->get_shape();
    const auto& C_layout = io_layouts[2];

    // We need find original M,N,K having layouts and ordered shapes
    // Layout:  0, 1, 2, 3   =>   New layout: 0, 2, 1, 3
    // Shape:   1, 3, 5, 9   =>   New Shape:  1, 5, 3, 9
    // To find original 2nd dimension, we should find index of position value `2` in new layout
    // and get dimension from new shape by this index
    auto get_ordered_idx = [](const std::vector<size_t>& layout, size_t idx) {
        return std::distance(layout.begin(), std::find(layout.begin(), layout.end(), idx));
    };

    m_K = A_shape[get_ordered_idx(A_layout, A_layout.size() - 1)];
    m_M = brgemm_node->get_input_count(0);
    m_N = C_shape[get_ordered_idx(C_layout, C_layout.size() - 1)];

    if (brgemm_node->is_with_data_repacking())
        leading_dimensions[1] = rnd_up(m_N, brgemm_copy->get_n_block_size());
    auto brg0Prc = brgemm_node->get_input_element_type(0);
    auto brg1Prc = brgemm_node->get_input_element_type(1);
    m_brg0VnniFactor = 4 / brg0Prc.size();
    bool brgWithAMX = brgemm_node->is_amx();

    io_data_size = {brg0Prc.size(), brg1Prc.size()};
    if (brgemm_node->get_input_size() == 3)
        io_data_size.push_back(brgemm_node->get_input_element_type(2).size());
    io_data_size.push_back(brgemm_node->get_output_element_type(0).size());

    m_with_comp = brgemm_node->is_with_compensations();
    m_with_scratch = brgemm_node->is_with_scratchpad();

    m_N_blk = brgemm_node->get_n_block_size();
    m_K_blk = brgemm_node->get_k_block_size();
    m_N_tail = m_N % m_N_blk;
    m_K_tail = m_K % m_K_blk;

    m_N_blk_loop = m_N >= 2 * m_N_blk;
    m_K_blk_loop = m_K >= 3 * m_K_blk;
    OPENVINO_ASSERT((!brgemm_node->is_with_data_repacking()) || (!m_N_blk_loop && !m_K_blk_loop),
                    "jit_brgemm_emitter doesn't support blocking by K, N dimensions when data repacking is needed!");

    auto N = [&](size_t n) {
        switch (n) {
            case 0: return m_N_blk;
            case 1: return m_N_tail;
            default: OPENVINO_THROW("jit_brgemm_emitter detected unsupported N value");
        }
    };
    auto K = [&](size_t k) {
        switch (k) {
            case 0: return m_K_blk;
            case 1: return m_K >= 2 * m_K_blk ? m_K_blk : 0;
            case 2: return m_K_tail;
            default:  OPENVINO_THROW("jit_brgemm_emitter detected unsupported K value");
        }
    };

    bool has_K_kernel = false;
    for (size_t k = 0; k < BRGEMM_K_KERNEL_NUM; k++) {
        bool has_N_kernel = false;
        for (size_t n = 0; n < BRGEMM_N_KERNEL_NUM; n++) {
            const size_t kernel_idx = getBrgIdx(k, n);
            auto& brgemmCtx = m_brgCtxs[kernel_idx];

            brgemmCtx.M = m_M;
            brgemmCtx.N = N(n);
            brgemmCtx.K = K(k);
            brgemmCtx.LDA = leading_dimensions[0];
            brgemmCtx.LDB = leading_dimensions[1];
            brgemmCtx.LDC = leading_dimensions[2];
            brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
            brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg1Prc));
            brgemmCtx.beta = has_K_kernel ? 1 : 0;

            if (brgemmCtx.N == 0 || brgemmCtx.N > m_N ||
                brgemmCtx.K == 0 || brgemmCtx.K > m_K)
                continue;

            initBrgemm(brgemmCtx, m_brgKernels[kernel_idx], brgWithAMX);
            has_N_kernel = true;
        }
        if (has_N_kernel)
            has_K_kernel = true;
    }

    m_load_offset_a = brgemm_node->get_offset_a();
    m_load_offset_b = brgemm_node->get_offset_b();
    m_store_offset_c = brgemm_node->get_offset_c();
    if (m_with_scratch)
        m_load_offset_scratch = brgemm_node->get_offset_scratch();
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OPENVINO_ASSERT(brgemm, "jit_brgemm_emitter::get_supported_precisions() expects BrgemmCPU node");
    switch (brgemm->get_type()) {
        case BrgemmCPU::Type::Floating:
            return {{element::f32, element::f32}};
        case BrgemmCPU::Type::WithDataRepacking:
            return {{element::u8, element::i8},
                    {element::bf16, element::bf16}};
        case BrgemmCPU::Type::WithCompensations:
            return {{element::i8, element::i8, element::f32}};
        case BrgemmCPU::Type::AMX:
            return {{element::i8, element::i8, element::u8},
                    {element::u8, element::i8, element::u8},
                    {element::bf16, element::bf16, element::u8}};
        default:
            OPENVINO_THROW("jit_brgemm_emitter got BrgemmCPU node with unsupported type");
    }
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    std::set<size_t> unique_ids{in[0], in[1], out[0]};
    size_t unique_ids_count = 3;
    auto add_reg_to_unique_ids = [&](const size_t reg_number) {
        unique_ids.insert(reg_number);
        unique_ids_count++;
    };

    if (m_N_blk_loop || m_K_blk_loop) {
        if (aux_gpr_idxs.size() < static_cast<size_t>(m_N_blk_loop) + static_cast<size_t>(m_K_blk_loop))
            OPENVINO_THROW("BRGEMM Emitter requires extra gpr which was not allocated");
        if (m_N_blk_loop)
            add_reg_to_unique_ids(aux_gpr_idxs[0]);
        if (m_K_blk_loop)
            add_reg_to_unique_ids(aux_gpr_idxs[m_N_blk_loop]);
    }
    if (m_with_scratch) {
        if (in.size() != 3)
            OPENVINO_THROW("BRGEMM Emitter expects 3 inputs if there are compensations/wsp");
        add_reg_to_unique_ids(in[2]);
    }
    if (unique_ids.size() != unique_ids_count) {
        OPENVINO_THROW("BRGEMM Emitter expects that all input/output registers are unique");
    }
}

void jit_brgemm_emitter::initBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, bool use_amx) {
    brgemm_t brgDesc;
    const bool is_int8 = utils::one_of(ctx.dt_in0, data_type::u8, data_type::s8) && utils::one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    auto isa = use_amx ? isa_undef
                       : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : (is_int8 ? avx512_core_vnni : avx512_core);
    auto status = brgemm_desc_init(&brgDesc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
                                   false, false, brgemm_row_major, 1.f, ctx.beta, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, nullptr);
    if (status != dnnl_success)
        OPENVINO_THROW("jit_brgemm_emitter cannot initialize brgemm descriptor due to invalid params");

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(brgDesc, ctx.palette);
    if (use_amx)
        amx_tile_configure(ctx.palette);

    ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success)
        OPENVINO_THROW("jit_brgemm_emitter cannot create brgemm kernel due to invalid params");
    brgKernel.reset(brgKernel_);
}

size_t jit_brgemm_emitter::aux_gprs_count() const {
    return m_N_blk_loop + m_K_blk_loop;
}

void jit_brgemm_emitter::emit_N_blocking_loops(size_t k_kernel_id,
                                          const Xbyak::Reg64& input_0, const Xbyak::Reg64& input_1,
                                          const Xbyak::Reg64& input_2, const Xbyak::Reg64& output_0,
                                          const Xbyak::Reg64& work_amount_N) const {
    // Blocked N loop
    size_t kernel_idx = getBrgIdx(k_kernel_id, 0);
    if (m_brgKernels[kernel_idx]) {
        const auto& brgemmCtx = m_brgCtxs[kernel_idx];
        Label N_loop_begin;
        if (m_N_blk_loop) {
            h->mov(work_amount_N, m_N);
            h->L(N_loop_begin);
        }

        emit_brgemm_kernel_call(m_brgKernels[kernel_idx].get(), brgemmCtx, input_0, input_1, input_2, output_0);
        // We don't need to increment pointers if we cover full N dimension in one kernel call
        if (m_N_blk_loop || m_N_tail != 0) {
            h->add(output_0, brgemmCtx.N * io_data_size.back());
            h->add(input_1, brgemmCtx.N * io_data_size[1]);
            if (m_with_scratch && m_with_comp)
                h->add(input_2, brgemmCtx.N * io_data_size[2]);
        }

        if (m_N_blk_loop) {
            h->sub(work_amount_N, brgemmCtx.N);
            h->cmp(work_amount_N, brgemmCtx.N);
            h->jge(N_loop_begin);
        }
    }
    // N loop tail
    kernel_idx = getBrgIdx(k_kernel_id, 1);
    if (m_brgKernels[kernel_idx])
        emit_brgemm_kernel_call(m_brgKernels[kernel_idx].get(), m_brgCtxs[kernel_idx], input_0, input_1, input_2, output_0);

    if (m_N_blk_loop || m_N_tail != 0) {
        h->sub(input_1, (m_N - m_N_tail) * io_data_size[1]);
        h->sub(output_0, (m_N - m_N_tail) * io_data_size.back());
        if (m_with_scratch && m_with_comp)
            h->sub(input_2, (m_N - m_N_tail) * io_data_size[2]);
    }
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 input_0(static_cast<int>(in[0]));
        Xbyak::Reg64 input_1(static_cast<int>(in[1]));
        Xbyak::Reg64 input_2(static_cast<int>(0));  // scratch. Default reg index is 0 if there isn't scratch
        Xbyak::Reg64 output_0(static_cast<int>(out[0]));
        Xbyak::Reg64 work_amount_N(m_N_blk_loop ? static_cast<int>(aux_gpr_idxs[0]) : 0);
        Xbyak::Reg64 work_amount_K(m_K_blk_loop ? static_cast<int>(aux_gpr_idxs[m_N_blk_loop]) : 0);
        h->add(input_0, m_load_offset_a);
        h->add(input_1, m_load_offset_b);
        h->add(output_0, m_store_offset_c);
        if (m_with_scratch) {
            input_2 = Xbyak::Reg64(static_cast<int>(in[2]));
            h->add(input_2, m_load_offset_scratch);
        }

        // fills kernel_idx with the first idx of non-empty K kernel or returns false
        auto get_K_kernel_idx = [&](size_t k_kernel_id, size_t& kernel_idx) {
            for (size_t n = 0; n < BRGEMM_N_KERNEL_NUM; n++) {
                const auto idx = getBrgIdx(k_kernel_id, n);
                if (m_brgKernels[idx]) {
                    kernel_idx = idx;
                    return true;
                }
            }
            return false;
        };
        // Blocked K loop
        const auto k_tail_id = BRGEMM_K_KERNEL_NUM - 1;
        size_t total_K_work_amount = m_K;
        size_t kernel_idx = SIZE_MAX;
        for (size_t k_blocked_id = 0; k_blocked_id < k_tail_id; k_blocked_id++) {
            if (get_K_kernel_idx(k_blocked_id, kernel_idx)) {
                const auto& brgemmCtx = m_brgCtxs[kernel_idx];
                Label K_loop_begin;
                // Note: we never emit loop for the first blocked kernel, since it always executed only once.
                // The purpose of the first blocked K kernel is to initializes output, because it has beta = 0
                if (k_blocked_id == 0) {
                    total_K_work_amount -= brgemmCtx.K;
                } else if (m_K_blk_loop) {
                    h->mov(work_amount_K, total_K_work_amount);
                    h->L(K_loop_begin);
                }

                emit_N_blocking_loops(k_blocked_id, input_0, input_1, input_2, output_0, work_amount_N);
                h->add(input_0, brgemmCtx.K * io_data_size[0]);
                h->add(input_1, (brgemmCtx.K * brgemmCtx.LDB) * io_data_size[1]);
                if (m_K_blk_loop && k_blocked_id) {
                    h->sub(work_amount_K, brgemmCtx.K);
                    h->cmp(work_amount_K, brgemmCtx.K);
                    h->jge(K_loop_begin);
                }
            }
        }
        // K loop tail
        if (get_K_kernel_idx(k_tail_id, kernel_idx)) {
            emit_N_blocking_loops(k_tail_id, input_0, input_1, input_2, output_0, work_amount_N);
        }

        h->sub(input_0, m_load_offset_a + (m_K - m_K_tail) * io_data_size[0]);
        h->sub(input_1, m_load_offset_b + (m_K - m_K_tail) * m_brgCtxs[0].LDB * io_data_size[1]);
        if (m_with_scratch)
            h->sub(input_2, m_load_offset_scratch);
        h->sub(output_0, m_store_offset_c);
    } else {
        OPENVINO_THROW("jit_brgemm_emitter requires at least avx512_core instruction set");
    }
}

void jit_brgemm_emitter::emit_brgemm_kernel_call(const brgemm_kernel_t *brg_kernel, const brgemmCtx& ctx,
                                            Reg64 addr_A, Reg64 addr_B, Reg64 scratch, Reg64 addr_C,
                                            const size_t in0_kernel_offset, const size_t in1_kernel_offset,
                                            const size_t in2_kernel_offset, const size_t out0_kernel_offset) const {
    constexpr size_t gpr_size = 8;
    if (ctx.is_with_amx) {
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                         h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // save function address in gpr to pass in call instruction
        const auto& overload = static_cast<status_t(*)(const char*)>(amx_tile_configure);
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(overload));
        h->mov(abi_param1, reinterpret_cast<uintptr_t>(ctx.palette));

        // align stack on 16-byte as ABI requires
        // note that RBX must not be changed by the callee
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        h->call(h->rbp);

        h->add(h->rsp, h->rbx);
        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);
    }

    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                     h->rax, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    // caller obligation to save k-regs as callee may use them
    size_t n_k_regs_to_save = 8;
    h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
    for (size_t i = 0; i < n_k_regs_to_save; ++i) {
        if (mayiuse(avx512_core))
            h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
        else
            h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
    }

    // 1. Caller obligation to save vector registers as callee may use them.
    // 2. There is an implicit assumption that the host code uses the same
    // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
    // `vlen` should be replaced with `host_isa::vlen` and
    // `host_isa::vecs_count`.
    h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
    for (size_t i = 0; i < get_max_vecs_count(); ++i)
        h->uni_vmovups(h->ptr[h->rsp + i * get_vec_length()], Zmm(i));

    // save function address in gpr to pass in call instruction
    const auto& brgemm_kernel_overload = static_cast<void (*)(const brgemm_kernel_t*,
                                                              const void*,
                                                              const void*,
                                                              void*,
                                                              void*,
                                                              int)>(kernel_execute);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(brgemm_kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), addr_A);
    h->uni_vmovq(Xmm(1), addr_B);
    h->uni_vmovq(Xmm(2), addr_C);
    if (m_with_scratch)
        h->uni_vmovq(Xmm(3), scratch);
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    const auto data_ptr_reg = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(brg_kernel));
    data_ptr_reg(Xmm(0), abi_param2, in0_kernel_offset);
    data_ptr_reg(Xmm(1), abi_param3, in1_kernel_offset);
    data_ptr_reg(Xmm(2), abi_param4, out0_kernel_offset);

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    const size_t num_args_passed_on_stack = 6;  // count of function brgemm_kernel_overload() parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);
    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);

    // Push the remaining parameters on the stack
    if (m_with_scratch) {
        h->uni_vmovq(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], Xmm(3));
        if (in2_kernel_offset) h->add(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], in2_kernel_offset);
    } else {
        h->mov(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_not_param1, static_cast<int>(m_with_comp));
    h->mov(h->qword[h->rsp + (abi_param_count + 1) * gpr_size], abi_not_param1);
#else
    if (m_with_scratch) {
        data_ptr_reg(Xmm(3), abi_param5, in2_kernel_offset);
    } else {
        h->mov(abi_param5, reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_param6, static_cast<int>(m_with_comp));
#endif

    // align stack on 16-byte as ABI requires
    // note that RBX must not be changed by the callee
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);

    h->call(h->rbp);

    h->add(h->rsp, h->rbx);

#ifdef _WIN32
    h->add(h->rsp, num_args_passed_on_stack * gpr_size);
#endif
    // restore vector registers
    for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
        h->uni_vmovups(Zmm(i), h->ptr[h->rsp + i * get_vec_length()]);
    }
    h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

    // restore k registers
    for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
        if (mayiuse(avx512_core))
            h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        else
            h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
    }
    h->add(h->rsp, n_k_regs_to_save * k_mask_size);

    // restore gpr registers
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}

void jit_brgemm_emitter::kernel_execute(const brgemm_kernel_t *brg_kernel,
                                   const void *A, const void *B, void *C, void *scratch, int with_comp) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = A;
    brgemm_p.ptr_B = B;
    brgemm_p.ptr_C = C;
    brgemm_p.ptr_D = C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = static_cast<size_t>(with_comp);
    brgemm_p.do_apply_comp = static_cast<size_t>(with_comp);
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

}   // namespace intel_cpu
}   // namespace ov
