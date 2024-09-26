// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_a.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

BrgemmCopyAKernelConfig::BrgemmCopyAKernelConfig(const element::Type& src_dt, cpu_isa_t isa)
    : m_static_params(std::make_shared<StaticParams>(src_dt, isa)) {
    m_hash = compute_hash();
}

bool BrgemmCopyAKernelConfig::is_completed() const {
    return !utils::one_of(0, m_curr_M_blk, m_K, m_copy_A_wei_stride, m_LDA) || is_empty();
}

bool BrgemmCopyAKernelConfig::is_empty() const {
    return everyone_is(0, m_curr_M_blk, m_K, m_copy_A_wei_stride, m_LDA);
}

bool BrgemmCopyAKernelConfig::operator==(const BrgemmCopyAKernelConfig& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(m_hash) && EQ(m_curr_M_blk) && EQ(m_K) && EQ(m_copy_A_wei_stride) && EQ(m_LDA) &&
           (EQ(m_static_params.get()) || *m_static_params == *(rhs.m_static_params));
#undef EQ
}

void BrgemmCopyAKernelConfig::update(dnnl_dim_t cur_M_blk, dnnl_dim_t K, dnnl_dim_t copy_A_wei_stride, dnnl_dim_t LDA) {
    // If one of the dims is zero, it means that BrgemmCopyB won't be executed (in Loop with work_amount = 0, for example)
    // To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (utils::one_of(0, cur_M_blk, K)) {
        m_curr_M_blk = 0; m_K = 0;
        m_copy_A_wei_stride = 0; m_LDA = 0;
    } else {
        m_curr_M_blk = cur_M_blk; m_K = K;
        m_copy_A_wei_stride = copy_A_wei_stride; m_LDA = LDA;
    }
    m_hash = compute_hash();
}

size_t BrgemmCopyAKernelConfig::compute_hash() const {
    size_t seed = m_static_params->hash;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(m_curr_M_blk); HASH(m_K);
    HASH(m_copy_A_wei_stride); HASH(m_LDA);
#undef HASH
    return seed;
}

BrgemmCopyAKernelConfig::StaticParams::StaticParams(const element::Type& etype, dnnl::impl::cpu::x64::cpu_isa_t isa)
    : src_dt(DTYPE_CAST(etype)), isa(isa), K_blk(brgemm_utils::repacking::compute_inner_k_block(etype)),
      vnni_factor(data_type_vnni_granularity(src_dt)), hash(init_hash(src_dt, isa, K_blk)) {}

bool BrgemmCopyAKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(hash) && EQ(src_dt) && EQ(isa) && EQ(K_blk);
#undef EQ
}

size_t BrgemmCopyAKernelConfig::StaticParams::init_hash(const dnnl_data_type_t& src_dt, dnnl::impl::cpu::x64::cpu_isa_t isa, dnnl_dim_t K_blk) {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(src_dt); HASH(isa); HASH(K_blk);
#undef HASH
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
#define PRINT(X) ss << #X  << " = " << X << "\n"
std::string BrgemmCopyAKernelConfig::to_string() const {
    std::stringstream ss;
    ss << m_static_params->to_string() << "\n";
    PRINT(m_hash); PRINT(m_curr_M_blk); PRINT(m_K);
    PRINT(m_copy_A_wei_stride); PRINT(m_LDA);
    return ss.str();
}
std::string BrgemmCopyAKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(src_dt); PRINT(isa); PRINT(K_blk);
    return ss.str();
}
#undef PRINT
#endif

BrgemmCopyAKernelExecutor::BrgemmCopyAKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmCopyAKernelConfig config)
    : CPUKernelExecutor<BrgemmCopyAKernelConfig, BrgemmCopyAKernel>(std::move(kernel_cache), std::move(config)) { }

std::shared_ptr<BrgemmCopyAKernel> BrgemmCopyAKernelExecutor::compile_kernel(const BrgemmCopyAKernelConfig& config) const {
    auto kernel = std::make_shared<BrgemmCopyAKernel>();

    // BrgemmCopyA is not executable - nothing to compile
    if (config.is_empty())
        return kernel;

    matmul::brgemm_matmul_conf_t conf;
    conf.src_tag = dnnl_abcd; // unused
    conf.K = config.get_K();
    conf.K_tail = config.get_K_tail();
    conf.K_blk = config.get_K_blk();
    conf.use_buffer_a_tail_only = false;
    //padding K tail to K_blk, LDA is the stride for target tensor
    conf.LDA = config.get_LDA();
    conf.has_zero_point_b = false;
    conf.s8s8_compensation_required = false;
    conf.wei_zp_type = dnnl::impl::cpu::x64::none;
    conf.src_zp_type = dnnl::impl::cpu::x64::none;
    conf.src_dt = config.get_src_dt();
    conf.copy_A_src_stride = config.get_copy_A_wei_stride();
    conf.a_dt_sz = dnnl_data_type_size(conf.src_dt);
    // copied A has the same precision of original
    conf.tr_a_dt_sz = dnnl_data_type_size(conf.src_dt);
    conf.transposed_A = false;
    conf.isa = config.get_isa();

    auto status = create_brgemm_matmul_copy_a(kernel->compiled_kernel, &conf);
    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot create brgemm copy a kernel due to invalid params");

    return kernel;
}

void BrgemmCopyAKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                              const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                              BrgemmCopyAKernelConfig& config) const {
    const auto& input_desc = expr->get_input_port_descriptor(0);
    const auto& output_desc = expr->get_output_port_descriptor(0);

    const auto planar_shape = ov::snippets::utils::get_planar_vdims(expr->get_input_port(0));
    const auto& in_subtensor = input_desc->get_subtensor();

    size_t loop_idx = 0;
    const auto& loop_ids = expr->get_loop_ids();
    const auto& loop_manager = linear_ir->get_loop_manager();

    auto get_blk = [&](size_t idx) {
        OPENVINO_ASSERT(idx < planar_shape.size() && idx < in_subtensor.size(), "Index must be less than shape/subtensor rank!");
        const auto  dim = *(planar_shape.rbegin() + idx);
        size_t blk = *(in_subtensor.rbegin() + idx);
        if (ov::snippets::utils::is_full_dim_value(blk)) {
            blk = dim;
        } else {
            OPENVINO_ASSERT(loop_idx < loop_ids.size(), "Loop is missed");
            const auto& current_expanded_loop_info = loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_ids[loop_idx++]);
            blk = current_expanded_loop_info->get_work_amount() > 0 ? current_expanded_loop_info->get_increment() : 0;
            input_desc->set_subtensor_dim(idx, blk);
            output_desc->set_subtensor_dim(idx, blk);
            OV_CPU_JIT_EMITTER_ASSERT(blk <= dim, "BrgemmCopyA has incompatible subtensor dimensions");
        }
        return blk;
    };

    //  Dimension M
    const size_t M_blk = get_blk(1);
    //  Dimension K
    const size_t K_blk = get_blk(0);

    const auto& src_type = expr->get_node()->get_input_element_type(0);
    const auto LDA = brgemm_utils::repacking::compute_LDA(K_blk, src_type);
    const auto copy_A_wei_stride = ov::snippets::utils::get_dim_stride(expr->get_input_port(0), 1) * src_type.size();

    config.update(M_blk, K_blk, copy_A_wei_stride, LDA);
}

void BrgemmCopyAKernelExecutor::execute(const BrgemmCopyAKernelExecutor* executor, call_args* args) {
    const auto& kernel = executor->get_kernel();
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr kernel");
    OV_CPU_JIT_EMITTER_ASSERT(args, "has nullptr call args");
    OV_CPU_JIT_EMITTER_ASSERT(kernel->compiled_kernel, "has nullptr kernel");
    const auto& config = static_cast<const BrgemmCopyAKernelConfig&>(executor->get_config());

    auto ctx = matmul::jit_brgemm_matmul_copy_a_t::ctx_t();

    ctx.current_M_blk = config.get_curr_M_blk();
    ctx.zp_b_compensation_buffer_ptr = nullptr;
    ctx.zp_a_compensation_result_ptr = nullptr;
    ctx.zp_b_neg_value_ptr = nullptr;
    ctx.zp_ab_comp_ptr = nullptr;

    const uint8_t* src = reinterpret_cast<const uint8_t*>(args->src);
    uint8_t* tr_src = reinterpret_cast<uint8_t*>(args->tr_src);

    size_t start_in = 0;
    size_t start_out = 0;

    const auto data_size = dnnl_data_type_size(config.get_src_dt());

    auto add_ptr_increments = [&](size_t current_K) {
        start_in += current_K * data_size;
        start_out += current_K * data_size;
    };

    const size_t block_count = config.get_K() / config.get_K_blk();
    for (size_t i = 0; i < block_count; ++i) {
        ctx.src = src + start_in;
        ctx.tr_src = tr_src + start_out;
        ctx.current_K_start = i * config.get_K_blk();
        ctx.current_K_blk = config.get_K_blk();

        (*kernel->compiled_kernel)(&ctx);

        add_ptr_increments(config.get_K_blk());
    }

    if (config.get_K_tail()) {
        ctx.src = src + start_in;
        ctx.tr_src = tr_src + start_out;
        ctx.current_K_start = block_count * config.get_K_blk();
        ctx.current_K_blk = config.get_K_tail();

        (*kernel->compiled_kernel)(&ctx);
    }
}

}   // namespace intel_cpu
}   // namespace ov
