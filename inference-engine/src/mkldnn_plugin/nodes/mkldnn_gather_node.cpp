// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_gather_node.h"
#include <ngraph/opsets/opset1.hpp>
#include <precision_utils.h>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGatherNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto gatherOp = ngraph::as_type_ptr<const ngraph::op::v7::Gather>(op);
        if (!gatherOp) {
            errorMessage = "Only opset7 Gather operation is supported";
            return false;
        }

        auto axesOp = gatherOp->get_input_node_shared_ptr(GATHER_AXIS);
        if (!ngraph::as_type_ptr<const ngraph::op::Constant>(axesOp)) {
            errorMessage = "Only Constant operation on 'axis' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNode::MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    errorPrefix_ = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto gatherOp = ngraph::as_type_ptr<ngraph::op::v7::Gather>(op);
    if (gatherOp->get_input_size() != 3 || gatherOp->get_output_size() != 1)
        IE_THROW() << errorPrefix_ << "has incorrect number of input/output edges!";

    const SizeVector& srcDims = gatherOp->get_input_shape(GATHER_DATA);
    const SizeVector& idxDims = gatherOp->get_input_shape(GATHER_INDEXES);
    const SizeVector& dstDims = gatherOp->get_output_shape(0);
    if (srcDims.size() == 0)
        IE_THROW() << errorPrefix_ << "has incorrect input parameters dimension!";

    axis = static_cast<int>(gatherOp->get_axis());
    if (axis < 0)
        axis += srcDims.size();
    if (!(0 <= axis && axis < static_cast<int>(srcDims.size())))
        IE_THROW() << errorPrefix_ << "has incorrect input parameters dimensions and axis number!";

    batchDims = static_cast<int>(gatherOp->get_batch_dims());
    if (batchDims < 0)
        batchDims += idxDims.size();
    if (!(0 <= batchDims && batchDims < std::min(static_cast<int>(srcDims.size()), static_cast<int>(idxDims.size()))) ||
        batchDims > axis)
        IE_THROW() << errorPrefix_ << "has incorrect batch_dims!";

    for (size_t i = 0; i < batchDims; i++) {
        if (srcDims[i] != idxDims[i])
            IE_THROW() << errorPrefix_ << "has incorrect first " << batchDims << " data and indices dimensions!";
    }

    indexRange = srcDims[axis];
    batchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
    outerSize = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
    dataLength = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<size_t>());
    srcBatchStride = std::accumulate(srcDims.begin() + batchDims, srcDims.end(), 1, std::multiplies<size_t>());
    idxBatchStride = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
    dstBatchStride = std::accumulate(dstDims.begin() + batchDims, dstDims.end(), 1, std::multiplies<size_t>());

    if (dataLength == 0)
        IE_THROW() << errorPrefix_ << "had incorrect input parameters dimension!";
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inIdxPrecision = getOriginalInputPrecisionAtPort(GATHER_INDEXES);
    if (inIdxPrecision != Precision::FP32 && inIdxPrecision != Precision::I32)
        inIdxPrecision = Precision::I32;

    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, dataPrecision},
                          {TensorDescCreatorTypes::ncsp, inIdxPrecision},
                          {TensorDescCreatorTypes::ncsp, Precision::I32}},
                         {{TensorDescCreatorTypes::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

template <typename index_t, class Conversion>
void MKLDNNGatherNode::gather() {
    const index_t* srcIndexes = reinterpret_cast<const index_t*>(getParentEdgeAt(GATHER_INDEXES)->getMemoryPtr()->GetPtr());
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const size_t dataSize = getSelectedPrimitiveDescriptor()->getConfig().inConfs[GATHER_DATA].desc.getPrecision().size();
    const size_t len = dataLength * dataSize;

    parallel_for2d(batchSize, idxBatchStride, [&](const size_t i, const size_t j) {
        int32_t idx = (srcIndexes[i * idxBatchStride + j]);
        if (idx < 0)
            idx += indexRange;

        for (size_t k = 0; k < outerSize; ++k) {
            const size_t srcStride = (i * srcBatchStride + k * dataLength * indexRange) * dataSize;
            const size_t dstStride = (i * dstBatchStride + k * dataLength * idxBatchStride) * dataSize;

            cpu_memcpy(&dstData[dstStride + j * len], &srcData[srcStride + idx * len], len);
        }
    });
}

void MKLDNNGatherNode::execute(mkldnn::stream strm) {
    switch (getParentEdgeAt(GATHER_INDEXES)->getDesc().getPrecision()) {
        case Precision::FP32:
            gather<float, f32toi32>();
            break;
        case Precision::I32:
            gather<int32_t, i32toi32>();
            break;
        default:
            return IE_THROW() << "Unsupported indices input precision";
    }
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
