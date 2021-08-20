// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matmul_node.h"

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "mkldnn_eltwise_node.h"

#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "mkldnn_extension_utils.h"

#include "emitters/jit_emitter.hpp"
#include "emitters/jit_eltwise_emitters.hpp"
#include "emitters/jit_mkldnn_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_matmul_args, field)


template <cpu_isa_t isa>
struct jit_uni_matmul_kernel_f32 : public jit_uni_matmul_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_matmul_kernel_f32)

    explicit jit_uni_matmul_kernel_f32(jit_matmul_config_params jcp_, const mkldnn_primitive_attr &attr) : jit_uni_matmul_kernel(jcp_, attr), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this,
                        post_op.eltwise.alg,
                        post_op.eltwise.alpha,
                        post_op.eltwise.beta,
                        1));
            } else {
                IE_THROW() << "MatMul supports only eltwise post ops!";
            }
        }

        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));

        this->preamble();

        mov(reg_src_a, ptr[reg_params + GET_OFF(src_a)]);
        mov(reg_src_b, ptr[reg_params + GET_OFF(src_b)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        amount_full = (jcp_.b_is_optimized ? jcp_.k : jcp_.n) / vec_step;
        amount_tail = (jcp_.b_is_optimized ? jcp_.k : jcp_.n) % vec_step;

        body();

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vec_step = vlen / sizeof(float);
    int amount_full;
    int amount_tail;
    int dst_count = 1;

    Vmm get_src_vmm(const int idx) {
        return Vmm(1 + dst_count + idx);
    }

    Vmm get_dst_vmm(const int idx = 0) {
        return Vmm(1 + idx);
    }

    Xmm get_aux_xmm(const int idx) {
        return Xmm(idx);
    }

    Xbyak::Reg64 reg_src_a = r8;
    Xbyak::Reg64 reg_src_b = r9;
    Xbyak::Reg64 reg_dst = r10;
    Xbyak::Reg64 reg_src_aux_a = r11;
    Xbyak::Reg64 reg_src_aux_b = r12;

    // indexes
    Xbyak::Reg64 m = rax;
    Xbyak::Reg64 k = rbx;
    Xbyak::Reg64 n = rdx;

    Xbyak::Reg64 reg_params = abi_param1; // RDI | RCX

    // loaders and stores
    Xbyak::Reg64 reg_load_store_mask = rsi;
    Xbyak::Reg64 reg_load_table = rbp;

    Vmm vmm_zero = Vmm(0);

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;

    std::vector<size_t> load_pool_gpr_idxs;
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;

    inline void body() {
        Xbyak::Label label_m;
        Xbyak::Label label_m_end;

        mov(m, 0);
        L(label_m); {
            cmp(m, jcp_.m);
            je(label_m_end, T_NEAR);

            if (jcp_.b_is_optimized) {
                optimized_body_loop();
            } else {
                body_loop();
            }

            add(reg_src_a, jcp_.k * sizeof(float));
            add(m, 1);
            jmp(label_m, T_NEAR);
        }
        L(label_m_end);
    }

    // common execution : broadcast(a) * vmm_b
    inline void body_loop() {
        Xbyak::Label label_k;
        Xbyak::Label label_k_end;

        dst_count = amount_full + static_cast<int>(amount_tail != 0);

        mov(reg_src_aux_a, reg_src_a);
        mov(reg_src_aux_b, reg_src_b);
        for (int i = 0; i < dst_count; ++i)
            uni_vpxor(get_dst_vmm(i), get_dst_vmm(i), get_dst_vmm(i));

        mov(k, 0);
        L(label_k); {
            cmp(k, jcp_.k);
            je(label_k_end, T_NEAR);

            uni_vbroadcastss(get_src_vmm(0), ptr[reg_src_aux_a]);  // src_a

            for (int i = 0; i < amount_full; ++i) {
                load_ptr(get_src_vmm(1), ptr[reg_src_aux_b + i * vlen]);  // src_b
                uni_vfmadd231ps(get_dst_vmm(i), get_src_vmm(0), get_src_vmm(1));  // broadcast(a) * vmm_b
            }
            add(reg_src_aux_b, amount_full * vlen);

            if (amount_tail != 0) {
                load(reg_src_aux_b, get_src_vmm(1), amount_tail);
                uni_vfmadd231ps(get_dst_vmm(amount_full), get_src_vmm(0), get_src_vmm(1));
                add(reg_src_aux_b, amount_tail * sizeof(float));
            }

            add(reg_src_aux_a, sizeof(float));
            add(k, 1);
            jmp(label_k, T_NEAR);
        }
        L(label_k_end);

        for (int i = 0; i < amount_full; ++i) {
            apply_post_ops(get_dst_vmm(i).getIdx());
            store_ptr(ptr[reg_dst + i * vlen], get_dst_vmm(i));
        }
        add(reg_dst, amount_full * vlen);

        if (amount_tail != 0) {
            apply_post_ops(get_dst_vmm(amount_full).getIdx());
            store(get_dst_vmm(amount_full), reg_dst, amount_tail);
            add(reg_dst, amount_tail * sizeof(float));
        }
    }

    // optimized execution for cases with transposed matrix b or k = 1
    inline void optimized_body_loop() {
        Xbyak::Label label_n;
        Xbyak::Label label_n_end;

        mov(reg_src_aux_b, reg_src_b);

        mov(n, 0);
        L(label_n); {
            cmp(n, jcp_.n);
            je(label_n_end, T_NEAR);

            mov(reg_src_aux_a, reg_src_a);
            uni_vpxor(get_dst_vmm(), get_dst_vmm(), get_dst_vmm());

            for (int i = 0; i < amount_full; ++i) {
                load_ptr(get_src_vmm(0), ptr[reg_src_aux_a + i * vlen]);  // src_a
                load_ptr(get_src_vmm(1), ptr[reg_src_aux_b + i * vlen]);  // src_b

                uni_vfmadd231ps(get_dst_vmm(), get_src_vmm(0), get_src_vmm(1));
            }
            add(reg_src_aux_b, amount_full * vlen);

            if (amount_tail != 0) {
                add(reg_src_aux_a, amount_full * vlen);
                load(reg_src_aux_a, get_src_vmm(0), amount_tail, true);
                load(reg_src_aux_b, get_src_vmm(1), amount_tail, true);
                uni_vfmadd231ps(get_dst_vmm(), get_src_vmm(0), get_src_vmm(1));

                add(reg_src_aux_b, amount_tail * sizeof(float));
            }

            // hsum
            if (isa == x64::avx512_common) {
                Xbyak::Zmm zmm_dst = Xbyak::Zmm(get_dst_vmm().getIdx());
                vextractf32x4(get_aux_xmm(2), zmm_dst, 0);
                vextractf32x4(get_aux_xmm(3), zmm_dst, 1);
                addps(get_aux_xmm(2), get_aux_xmm(3));
                vextractf32x4(get_aux_xmm(3), zmm_dst, 2);
                vextractf32x4(get_aux_xmm(4), zmm_dst, 3);
                addps(get_aux_xmm(3), get_aux_xmm(4));
                vaddps(get_aux_xmm(1), get_aux_xmm(2), get_aux_xmm(3));
                hsum(get_aux_xmm(1));
            } else if (isa == x64::avx2) {
                Xbyak::Ymm ymm_dst = Xbyak::Ymm(get_dst_vmm().getIdx());
                vextractf128(get_aux_xmm(2), ymm_dst, 0);
                vextractf128(get_aux_xmm(3), ymm_dst, 1);
                vaddps(get_aux_xmm(1), get_aux_xmm(2), get_aux_xmm(3));
                hsum(get_aux_xmm(1));
            } else {
                hsum(get_dst_vmm());
            }

            apply_post_ops();

            store(get_dst_vmm(), reg_dst, 1);

            add(reg_dst, sizeof(float));

            add(n, 1);
            jmp(label_n, T_NEAR);
        }
        L(label_n_end);
    }

    inline void hsum(Xbyak::Xmm xmm) {
        movshdup(get_aux_xmm(2), xmm);
        addps(xmm, get_aux_xmm(2));
        movhlps(get_aux_xmm(2), xmm);
        addps(xmm, get_aux_xmm(2));
    }

    inline void apply_post_ops(const int idx = 1) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx++]->compute_vector_range(idx, idx + 1);
            }
        }
    }

    inline void load(Xbyak::Reg64 reg, Vmm vmm, int load_num, bool is_fill = false) {
        load_emitter->emit_code({static_cast<size_t>(reg.getIdx())}, {static_cast<size_t>(vmm.getIdx())},
                                std::make_shared<load_emitter_context>(Precision::FP32, Precision::FP32, load_num, 0, is_fill),
                                {}, load_pool_gpr_idxs);
    }

    inline void store(Vmm vmm, Xbyak::Reg64 reg, int load_num) {
        store_emitter->emit_code({static_cast<size_t>(vmm.getIdx())}, {static_cast<size_t>(reg.getIdx())},
                                 std::make_shared<store_emitter_context>(Precision::FP32, Precision::FP32, load_num),
                                 store_pool_vec_idxs, store_pool_gpr_idxs);
    }

    inline void load_ptr(Vmm vmm_src, const Xbyak::Address &op) {
        uni_vmovups(vmm_src, op);
    }

    inline void store_ptr(const Xbyak::Address &op, Vmm vmm_dst) {
        uni_vmovups(op, vmm_dst);
    }
};

bool MKLDNNMatMulNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }

        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        const auto shapeA = matMul->get_input_shape(0);
        const auto shapeB = matMul->get_input_shape(1);

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_shape(i).size();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_shape().size();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMatMulNode::MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
    MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    errorPrefix = "MatMul node with name '" + getName() + "'";

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();
}

bool MKLDNNMatMulNode::canFuse(const MKLDNNNodePtr& node) const {
    return one_of(node->getAlgorithm(), EltwiseRelu, EltwiseGelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseTanh,
                  EltwiseSwish, EltwiseHswish, EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven,
                  EltwiseRoundHalfAwayFromZero, EltwiseAbs, EltwiseSqrt, EltwiseSoftRelu);
}

void MKLDNNMatMulNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) const {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            eltwiseNode->appendPostOps(ops);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

std::shared_ptr<mkldnn::primitive_attr> MKLDNNMatMulNode::initPrimitiveAttr() const {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, true);

    return attr;
}

void MKLDNNMatMulNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    auto firstInPortPrec = getOriginalInputPrecisionAtPort(0);
    auto secondInPortPrec = getOriginalInputPrecisionAtPort(1);
    auto outPortPrec = getOriginalOutputPrecisionAtPort(0);

    if (firstInPortPrec.size() != secondInPortPrec.size())
        firstInPortPrec = secondInPortPrec = getMaxPrecision(getOriginalInputPrecisions());

    if (!fusedWith.empty()) {
        outPortPrec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (inputShapes[0].getRank() != inputShapes[1].getRank() || inputShapes[0].getRank() != outputShapes[0].getRank())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    const int nDims = inputShapes[0].getRank();
    const auto xAxis = nDims - 1;
    const auto yAxis = nDims - 2;
    const auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    const auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    const auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    const auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    const auto& inDims0 = getInputShapeAtPort(0).getStaticDims();
    const auto& inDims1 = getInputShapeAtPort(1).getStaticDims();
    const auto& outDims = getOutputShapeAtPort(0).getStaticDims();

    // coverity[copy_paste_error]
    if (inDims0[xAxis0] != inDims1[yAxis1] ||
        inDims0[yAxis0] != outDims[yAxis] ||
        inDims1[xAxis1] != outDims[xAxis])
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((inDims0[dim_idx] != outDims[dim_idx] &&
             inDims0[dim_idx] != 1) ||
            (inDims1[dim_idx] != outDims[dim_idx] &&
             inDims1[dim_idx] != 1)) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    /* Example MatMul:
     * 2x128x512(T) * 2x128x512 = 2x512x512
     * First input 2x128x512(T) should be transposed
     * oneDNN requires memory::desc for this input to:
     * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
     * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
     */
    auto getStridesAndDims = [](Shape& shape, const bool transpose) {
        const auto getRank = shape.getRank();

        VectorDims strides(getRank, 1);
        for (size_t i = 1; i < getRank; i++) {
            strides[getRank - i - 1 ] = strides[getRank - i] * shape.getStaticDims()[getRank - i];
        }

        if (transpose && getRank > 1) {
            // form new shape
            auto dims = shape.getStaticDims();
            std::swap(dims[getRank - 2], dims[getRank - 1]);
            shape = Shape{dims};
            // update strides
            strides[getRank - 1] = shape.getStaticDims()[getRank - 2];
            strides[getRank - 2] = 1;
        }

        return strides;
    };

    initialInShapes[0] = inputShapes[0];
    initialInShapes[1] = inputShapes[1];

    const VectorDims inStrides0 = getStridesAndDims(inputShapes[0], transposeIn[0]);
    const VectorDims inStrides1 = getStridesAndDims(inputShapes[1], transposeIn[1]);
    const VectorDims outStrides = getStridesAndDims(outputShapes[0], false);

    inDataDesc[0] = std::make_shared<DnnlBlockedMemoryDesc>(firstInPortPrec, inputShapes[0], inStrides0);
    inDataDesc[1] = std::make_shared<DnnlBlockedMemoryDesc>(secondInPortPrec, inputShapes[1], inStrides1);
    outDataDesc   = std::make_shared<DnnlBlockedMemoryDesc>(outPortPrec, getOutputShapeAtPort(0), outStrides);

    createDescriptor({inDataDesc[0], inDataDesc[1]}, {outDataDesc});
}

void MKLDNNMatMulNode::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                        const std::vector<MemoryDescPtr>& outputDesc) {
    MKLDNNDescriptor desc{
        std::shared_ptr<matmul::desc>(
            new matmul::desc(MemoryDescUtils::convertToDnnlMemoryDesc(inDataDesc[0])->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(inDataDesc[1])->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(outDataDesc)->getDnnlDesc()))};

    descs.push_back(desc);
}

void MKLDNNMatMulNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto attr = initPrimitiveAttr();

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), *attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = -1;
                portConfig.constant = false;

                auto src_desc = getSrcMemDesc(itpd, i);
                if (src_desc->getType() & MemoryDescType::Blocked) {
                    portConfig.desc = src_desc->as<BlockedMemoryDesc>()->cloneWithUndefStridesAndOffset();
                } else {
                    portConfig.desc = std::move(src_desc);
                }

                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = canBeInPlace() ? 0 : -1;
                portConfig.constant = false;

                auto dst_desc = getDstMemDesc(itpd, i);
                if (dst_desc->getType() & MemoryDescType::Blocked) {
                    portConfig.desc = dst_desc->as<BlockedMemoryDesc>()->cloneWithUndefStridesAndOffset();
                } else {
                    portConfig.desc = std::move(dst_desc);
                }

                config.outConfs.push_back(portConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void MKLDNNMatMulNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->GetPrimitivePtr() || !src1MemPtr || !src1MemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    const int inNDims0 = inputShapes[0].getRank();
    const int inNDims1 = inputShapes[1].getRank();
    const int outNDims = inputShapes[0].getRank();
    const int m = inputShapes[0].getStaticDims()[inNDims0 - 2];
    const int k = inputShapes[0].getStaticDims()[inNDims0 - 1];
    const int n = outputShapes[0].getStaticDims()[outNDims - 1];
    const auto precision = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision();

    bool canUseOptimizedExecution = inNDims0 == inNDims1 && inNDims0 == 2 && !transposeIn[0] &&
                                    m <= 128 && k <= 128 && precision == Precision::FP32;

    // to have enough vmm for unrolling by n for first algorithm with broadcast(a)
    if (canUseOptimizedExecution && !transposeIn[1] && n != 1) {
        const int nofree_registers = 3;  // vmm_zero, vmm_src_a, vmm_src_b
        int size = 1;
        int vmm_count = 16;

        if (mayiuse(impl::cpu::x64::avx512_common)) {
            size = 16;
            vmm_count = 32;
        } else if (mayiuse(cpu::x64::avx2)) {
            size = 8;
        } else if (mayiuse(cpu::x64::sse41)) {
            size = 4;
        }

        canUseOptimizedExecution = n <= ((vmm_count - nofree_registers) * size);
    }
    if (canUseOptimizedExecution) {
        jit_matmul_config_params jep;
        jep.m = m;
        jep.k = k;
        jep.n = n;
        jep.b_is_optimized = transposeIn[1] || jep.n == 1;

        arg = jit_matmul_args();
        memSrcA = getParentEdgeAt(0)->getMemoryPtr();
        memSrcB = getParentEdgeAt(1)->getMemoryPtr();
        memDst = getChildEdgeAt(0)->getMemoryPtr();

        if (mayiuse(x64::avx512_common)) {
            matmul_kernel.reset(new jit_uni_matmul_kernel_f32<x64::avx512_common>(jep, *attr.get()));
        } else if (mayiuse(x64::avx2)) {
            matmul_kernel.reset(new jit_uni_matmul_kernel_f32<x64::avx2>(jep, *attr.get()));
        } else if (mayiuse(x64::sse41)) {
            matmul_kernel.reset(new jit_uni_matmul_kernel_f32<x64::sse41>(jep, *attr.get()));
        }

        if (matmul_kernel)
            matmul_kernel->create_ker();
    }

    if (prim || matmul_kernel)
        return;

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();
    std::shared_ptr<matmul::primitive_desc> prim_desc;
    prim_desc = std::make_shared<matmul::primitive_desc>(
            createPrimitiveDescriptor<matmul::primitive_desc, matmul::desc>(*attr));

    prim.reset(new matmul(*prim_desc));

    auto src0 = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto src1 = getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();

    primArgs = {{DNNL_ARG_SRC_0, src0}, {DNNL_ARG_WEIGHTS_0, src1}, {DNNL_ARG_DST, dst}};
}

void MKLDNNMatMulNode::execute(mkldnn::stream strm) {
    if (matmul_kernel) {
        arg.src_a = memSrcA->GetPtr();
        arg.src_b = memSrcB->GetPtr();
        arg.dst   = memDst->GetPtr();

        (*matmul_kernel)(&arg);
        return;
    }

    MKLDNNNode::execute(strm);
}

MemoryDescPtr MKLDNNMatMulNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1): primitive_desc_it.src_desc(idx);

    return std::make_shared<CpuBlockedMemoryDesc>(
        MKLDNNExtensionUtils::DataTypeToIEPrecision(static_cast<mkldnn::memory::data_type>(desc.data.data_type)),
        initialInShapes[idx]); /* provide initial shapes, so hide transpose effect */
}

bool MKLDNNMatMulNode::created() const {
    return getType() == MatMul;
}

size_t MKLDNNMatMulNode::getMaxBatch() const {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MKLDNNMatMulNode::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
