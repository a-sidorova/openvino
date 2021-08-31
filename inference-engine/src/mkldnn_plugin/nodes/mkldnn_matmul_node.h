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
    size_t m;
    size_t n;
    size_t k;
    bool b_is_optimized;
};

struct jit_matmul_args {
    void *src_A;
    void *src_B;
    void *dst;
};

class MKLDNNMatMulNode;

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
    void createDescriptor(const std::vector<const MemoryDesc*>& inputDesc,
                          const std::vector<const MemoryDesc*>& outputDesc) override;
    void initSupportedPrimitiveDescriptors() override;
    std::unique_ptr<MKLDNNMemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    bool created() const override;
    int getMaxBatch() override;
    InferenceEngine::Precision getRuntimePrecision() const override;
    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return getOriginalInputsNumber();
    }

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights) const;

    std::string errorPrefix;

    /* whether to transpose input */
    std::array<bool, 2> transposeIn;

    std::array<std::unique_ptr<MKLDNNMemoryDesc>, 2> inDataDesc;
    std::unique_ptr<MKLDNNMemoryDesc> outDataDesc;

    MKLDNNMemoryPtr memSrcA;
    MKLDNNMemoryPtr memSrcB;
    MKLDNNMemoryPtr memDst;

    jit_matmul_args arg;
    mkldnn::primitive_attr attr;
    std::shared_ptr<jit_uni_matmul_kernel> matmul_kernel = nullptr;
};

}  // namespace MKLDNNPlugin

