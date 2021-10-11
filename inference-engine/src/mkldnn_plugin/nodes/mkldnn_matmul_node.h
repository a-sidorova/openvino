// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <ie_common.h>
#include <string>
#include <vector>
#include <array>

namespace MKLDNNPlugin {

struct jit_matmul_config_params {
    size_t b;
    size_t m;
    size_t n;
    size_t k;
    size_t stride0;
    size_t stride1;
    bool b_is_optimized;
};

struct jit_matmul_args {
    void *src_a;
    void *src_b;
    void *dst;
};

struct jit_uni_matmul_kernel {
    void (*ker_)(const jit_matmul_args *);

    void operator()(const jit_matmul_args *args) {
       assert(ker_);
       ker_(args);
   }

   explicit jit_uni_matmul_kernel(jit_matmul_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
   virtual ~jit_uni_matmul_kernel() {}

   virtual void create_ker() = 0;

    jit_matmul_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};

class MKLDNNMatMulNode : public MKLDNNNode {
public:
    MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void initSupportedPrimitiveDescriptors() override;
    MemoryDescPtr getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    bool created() const override;
    size_t getMaxBatch() const override;

    InferenceEngine::Precision getRuntimePrecision() const override;
    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return getOriginalInputsNumber();
    }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr() const override;

private:
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights) const;

    std::string errorPrefix;

    /* whether to transpose input */
    std::array<bool, 2> transposeIn;
    /* initial shapes without transpose,
     * necessary to hide transpose effect from plugin */
    std::array<Shape, 2> initialInShapes;

    std::array<MemoryDescPtr, 2> inDataDesc;
    MemoryDescPtr outDataDesc;

    /* custom matmul */
    MKLDNNMemoryPtr memSrcA = nullptr;
    MKLDNNMemoryPtr memSrcB = nullptr;
    MKLDNNMemoryPtr memDst = nullptr;
    jit_matmul_args arg;
    mkldnn::primitive_attr attr;
    std::shared_ptr<jit_uni_matmul_kernel> matmul_kernel = nullptr;
};

}  // namespace MKLDNNPlugin

