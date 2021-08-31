// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matmul_node.h"

#include "cpu_blocked_memory_desc.h"
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
#include "utils/general_utils.h"
#include "cpu_memory_desc_utils.h"

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

        mov(reg_src_a, ptr[reg_params + GET_OFF(src_A)]);
        mov(reg_src_b, ptr[reg_params + GET_OFF(src_B)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        amount_full = (jcp_.b_is_optimized ? jcp_.n : jcp_.k) / step;
        amount_tail = (jcp_.b_is_optimized ? jcp_.n : jcp_.k) % step;

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
    const int step = vlen / sizeof(float);
    int amount_full;
    int amount_tail;

    Reg64 get_aux_reg(const int idx) {
        return Reg64(r12.getIdx() + idx);
    }

    Vmm get_aux_vmm(const int idx) {
        return Vmm(3 + idx);
    }

    Xbyak::Reg64 reg_src_a = r8;
    Xbyak::Reg64 reg_src_b = r9;
    Xbyak::Reg64 reg_dst = r10;
    Xbyak::Reg64 reg_src_aux_a = r11;
    Xbyak::Reg64 reg_src_aux_b = r12;
    Xbyak::Reg64 reg_temp = rbp;

    // indexes
    Xbyak::Reg64 m = rax;
    Xbyak::Reg64 n = rbx;
    Xbyak::Reg64 k = rcx;

    Xbyak::Reg64 reg_params = abi_param1; // RDI | RCX

    // loaders and stores
    Xbyak::Reg64 reg_load_store_mask = rsi;
    Xbyak::Reg64 reg_load_table = rdi;

    Vmm vmm_zero = Vmm(0);
    Vmm vmm_dst = Vmm(1);
    Vmm vmm_a = Vmm(2);
    Vmm vmm_b = Vmm(3);

    Xmm xmm_aux1 = Xmm(1);
    Xmm xmm_aux2 = Xmm(2);
    Xmm xmm_aux3 = Xmm(3);

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

            mov(k, 0);
            if (jcp_.b_is_optimized) {
                optimized_body_loop();
            } else {
                body_loop();
                if (amount_tail != 0) {
                    body_loop(true);
                }
            }

            add(reg_src_a, jcp_.n * sizeof(float));
            add(m, 1);
            jmp(label_m, T_NEAR);
        }
        L(label_m_end);
    }

    // common execution : broadcast(a) * vmm_b
    inline void body_loop(bool is_tail = false) {
        Xbyak::Label label_k;
        Xbyak::Label label_k_end;
        Xbyak::Label label_n;
        Xbyak::Label label_n_end;

        int elt_num = is_tail ? amount_tail : step;
        int amount =  is_tail ? amount_full + 1 : amount_full;
        int unroll_count = !is_tail && amount_full % 2 == 0 ? 2 : 1;

        L(label_k); {
            cmp(k, amount);
            je(label_k_end, T_NEAR);

            for (int i = 0; i < unroll_count; i++)
                uni_vpxor(get_aux_vmm(unroll_count + i), get_aux_vmm(unroll_count + i), get_aux_vmm(unroll_count + i));

            mov(reg_src_aux_a, reg_src_a);
            mov(reg_src_aux_b, reg_src_b);
            imul(reg_temp, k, vlen);
            add(reg_src_aux_b, reg_temp);
            for (int i = 1; i < unroll_count; i++)
                mov(get_aux_reg(i), reg_src_aux_b);
            for (int i = 1; i < unroll_count; i++)
                add(get_aux_reg(i), vlen * i);

            mov(n, 0);
            L(label_n); {
                cmp(n, jcp_.n);
                je(label_n_end);

                uni_vbroadcastss(vmm_a, ptr[reg_src_aux_a]);
                for (int i = 0; i < unroll_count; i++)
                    load(get_aux_reg(i), get_aux_vmm(i), elt_num, is_tail);

                for (int i = 0; i < unroll_count; i++)
                    uni_vfmadd231ps(get_aux_vmm(unroll_count + i), vmm_a, get_aux_vmm(i));

                add(reg_src_aux_a, sizeof(float));
                for (int i = 0; i < unroll_count; i++)
                    add(get_aux_reg(i), jcp_.k * sizeof(float));

                add(n, 1);
                jmp(label_n);
            }
            L(label_n_end);

            for (int i = 0; i < unroll_count; i++)
                apply_post_ops(unroll_count + i);

            for (int i = 0; i < unroll_count; i++) {
                store(get_aux_vmm(unroll_count + i), reg_dst, elt_num);
                add(reg_dst, elt_num * sizeof(float));
            }

            add(k, unroll_count);
            jmp(label_k, T_NEAR);
        }
        L(label_k_end);
    }

    // optimized execution for cases with transposed b or k = 1
    inline void optimized_body_loop() {
        Xbyak::Label label_k;
        Xbyak::Label label_k_end;

        mov(reg_src_aux_b, reg_src_b);
        L(label_k); {
            cmp(k, jcp_.k);
            je(label_k_end, T_NEAR);

            mov(reg_src_aux_a, reg_src_a);
            uni_vpxor(vmm_dst, vmm_dst, vmm_dst);

            mov(n, 0);
            optimized_body_internal_loop_by_n();
            if (amount_tail != 0) {
                optimized_body_internal_loop_by_n(true);
            }

            // hsum
            if (isa == x64::avx512_common) {
                Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
                vextractf32x4(xmm_aux2, zmm_dst, 0);
                vextractf32x4(xmm_aux3, zmm_dst, 1);
                addps(xmm_aux2, xmm_aux3);
                vextractf32x4(xmm_aux3, zmm_dst, 2);
                vextractf32x4(xmm_aux1, zmm_dst, 3);
                addps(xmm_aux3, xmm_aux1);
                addps(xmm_aux1, xmm_aux3);
                hsum(xmm_aux1);
            } else if (isa == x64::avx2) {
                Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
                vextractf128(xmm_aux2, ymm_dst, 0);
                vextractf128(xmm_aux3, ymm_dst, 1);
                vaddps(xmm_aux1, xmm_aux2, xmm_aux3);
                hsum(xmm_aux1);
            } else {
                hsum(vmm_dst);
            }

            apply_post_ops();

            store(vmm_dst, reg_dst, 1);

            add(reg_dst, sizeof(float));

            add(k, 1);
            jmp(label_k, T_NEAR);
        }
        L(label_k_end);
    }

    inline void optimized_body_internal_loop_by_n(bool is_tail = false) {
        Xbyak::Label label_n;
        Xbyak::Label label_n_end;

        int elt_num = is_tail ? amount_tail : step;
        int amount =  is_tail ? amount_full + 1 : amount_full;

        L(label_n); {
            cmp(n, amount);
            je(label_n_end, T_NEAR);

            load(reg_src_aux_a, vmm_a, elt_num, is_tail);
            load(reg_src_aux_b, vmm_b, elt_num, is_tail);

            uni_vfmadd231ps(vmm_dst, vmm_a, vmm_b);

            add(reg_src_aux_a, elt_num * sizeof(float));
            add(reg_src_aux_b, elt_num * sizeof(float));

            add(n, 1);
            jmp(label_n, T_NEAR);
        }
        L(label_n_end);
    }

    inline void hsum(Xbyak::Xmm xmm) {
        movshdup(xmm_aux2, xmm);
        addps(xmm, xmm_aux2);
        movhlps(xmm_aux2, xmm);
        addps(xmm, xmm_aux2);
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
};

bool MKLDNNMatMulNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        const auto shapeA = matMul->get_input_shape(0);
        const auto shapeB = matMul->get_input_shape(1);

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_shape(i).size();
            if (inShapeRank < 2 || inShapeRank > 3) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_shape().size();
        if (outShapeRank < 2 || outShapeRank > 3) {
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

    const auto firstInDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(firstInPortPrec);
    const auto secondInDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(secondInPortPrec);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outPortPrec);

    if (!fusedWith.empty()) {
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }

    inputShapes.reserve(2);
    inputShapes[0] = getParentEdgeAt(0)->getShape();
    inputShapes[1] = getParentEdgeAt(1)->getShape();
    outputShapes[0] = getChildEdgeAt(0)->getShape();

    if (inputShapes[0].getRank() != inputShapes[1].getRank() || inputShapes[0].getRank() != outputShapes[0].getRank())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    const int nDims = inputShapes[0].getRank();
    const auto xAxis = nDims - 1;
    const auto yAxis = nDims - 2;
    const auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    const auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    const auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    const auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    const auto& inDims0 = inputShapes[0].getStaticDims();
    const auto& inDims1 = inputShapes[1].getStaticDims();
    const auto& outDims = outputShapes[0].getStaticDims();

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

        std::vector<size_t> strides(getRank, 1);
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

    const std::vector<size_t> inStrides0 = getStridesAndDims(inputShapes[0], transposeIn[0]);
    const std::vector<size_t> inStrides1 = getStridesAndDims(inputShapes[1], transposeIn[1]);
    const std::vector<size_t> outStrides = getStridesAndDims(outputShapes[0], false);

    inDataDesc[0] = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inDims0, firstInDataType, inStrides0);
    inDataDesc[1] = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(inDims1, secondInDataType, inStrides1);
    outDataDesc   = MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(outDims, outputDataType, outStrides);

    createDescriptor({inDataDesc[0].get(), inDataDesc[1].get()}, {outDataDesc.get()});
}

void MKLDNNMatMulNode::createDescriptor(const std::vector<const MemoryDesc*>& inputDesc,
                                        const std::vector<const MemoryDesc*>& outputDesc) {
    MKLDNNDescriptor desc{
        std::shared_ptr<matmul::desc>(
            new matmul::desc(*inDataDesc[0], *inDataDesc[1], *outDataDesc))};

    descs.push_back(desc);
}

void MKLDNNMatMulNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    setPostOps(attr, true);

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = -1;
                portConfig.constant = false;
                portConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getSrcMemDesc(itpd, i));
                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = canBeInPlace() ? 0 : -1;
                portConfig.constant = false;
                portConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getDstMemDesc(itpd, i));
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
    const int n = inputShapes[0].getStaticDims()[inNDims0 - 1];
    const int k = outputShapes[0].getStaticDims()[outNDims - 1];

    bool canUseOptimizedExecution = inNDims0 == inNDims1 && inNDims0 == 2 && !transposeIn[0] &&
                                    m <= 128 && n <= 128 && k <= 128;
    if (canUseOptimizedExecution) {
        jit_matmul_config_params jep;
        jep.m = m;
        jep.n = n;
        jep.k = k;
        jep.b_is_optimized = transposeIn[1] || jep.k == 1;

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

    std::shared_ptr<matmul::primitive_desc> prim_desc;
    prim_desc = std::make_shared<matmul::primitive_desc>(
            createPrimitiveDescriptor<matmul::primitive_desc, matmul::desc>(attr));

    prim.reset(new matmul(*prim_desc));

    auto src0 = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto src1 = getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();

    primArgs = {{DNNL_ARG_SRC_0, src0}, {DNNL_ARG_WEIGHTS_0, src1}, {DNNL_ARG_DST, dst}};
}

void MKLDNNMatMulNode::execute(mkldnn::stream strm) {
    if (matmul_kernel) {
        arg.src_A = memSrcA->GetPtr();
        arg.src_B = memSrcB->GetPtr();
        arg.dst   = memDst->GetPtr();

        (*matmul_kernel)(&arg);
        return;
    }

    MKLDNNNode::execute(strm);
}

std::unique_ptr<MKLDNNMemoryDesc> MKLDNNMatMulNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
        : MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

    auto parentShape = getParentEdgeAt(idx)->getShape();

    return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(MKLDNNMemoryDesc(getParentEdgeAt(idx)->getShape().getStaticDims(),
                                                                        MKLDNNExtensionUtils::IEPrecisionToDataType(desc.getPrecision()),
                                                                        MKLDNNMemory::GetPlainFormatByRank(getParentEdgeAt(idx)->getShape().getRank())));
}

bool MKLDNNMatMulNode::created() const {
    return getType() == MatMul;
}

int MKLDNNMatMulNode::getMaxBatch() {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MKLDNNMatMulNode::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
