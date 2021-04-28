// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGatherNode : public MKLDNNNode {
public:
    MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNGatherNode() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    struct f32toi32 {
        inline int32_t operator()(const float value) {
            return static_cast<int32_t>(value);
        }
    };

    struct i32toi32 {
        inline int32_t operator()(const int32_t value) {
            return value;
        }
    };

    int axis = 0;
    int batchDims = 0;

    size_t indexRange = 0;
    size_t batchSize = 1;
    size_t outerSize = 1;
    size_t dataLength = 1;
    size_t srcBatchStride = 1;
    size_t idxBatchStride = 1;
    size_t dstBatchStride = 1;

    static const size_t GATHER_DATA = 0;
    static const size_t GATHER_INDEXES = 1;
    static const size_t GATHER_AXIS = 2;

    std::string errorPrefix_;

    template <typename index_t, class Conversion>
    void gather();
};

}  // namespace MKLDNNPlugin
