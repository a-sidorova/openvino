// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pad_node.h"
#include <legacy/ie_layers.h>
#include <string>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPadNode::MKLDNNPadNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNPadNode::getSupportedDescriptors() {
    auto* padLayer = dynamic_cast<PadLayer*>(getCnnLayer().get());
    if (padLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert Pad layer.";

    padsBegin = padLayer->GetParamAsUInts("pads_begin");
    padsEnd = padLayer->GetParamAsUInts("pads_end");

    SizeVector srcDims = padLayer->insData[0].lock()->getTensorDesc().getDims();
    SizeVector dstDims = padLayer->outData[0]->getTensorDesc().getDims();
    if (srcDims.size() != dstDims.size() || padsBegin.size() != srcDims.size() || padsEnd.size() != srcDims.size())
        THROW_IE_EXCEPTION << padLayer->name << " Incorrect number of input/output dimensions!";

    std::string pad_mode = padLayer->GetParamAsString("pad_mode");
    if (pad_mode == "constant") {
        padMode = CONSTANT;
        padValue = padLayer->GetParamAsFloat("pad_value");
    } else if (pad_mode == "edge") {
        padMode = EDGE;
    } else if (pad_mode == "reflect") {
        padMode = REFLECT;
        for (size_t i = 0; i < srcDims.size(); i++) {
            if ((srcDims[i] - 1) < padsBegin[i] || (srcDims[i] - 1) < padsEnd[i])
                THROW_IE_EXCEPTION << padLayer->name << " Incorrect padsBegin or padsEnd for 'reflect' pad mode";
        }
    } else if (pad_mode == "symmetric") {
        padMode = SYMMETRIC;
        for (size_t i = 0; i < srcDims.size(); i++) {
            if (srcDims[i] < padsBegin[i] || srcDims[i] < padsEnd[i])
                THROW_IE_EXCEPTION << padLayer->name << " Incorrect padsBegin or padsEnd for 'symmetric' pad mode";
        }
    } else {
        THROW_IE_EXCEPTION << padLayer->name
                           << " Incorrect pad_mode. Only constants|edge|reflect|symmetric modes are supported!";
    }

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNPadNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    auto memoryFormat = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memoryFormat);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memoryFormat);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memoryFormat});

    if (getParentEdgeAt(0)->getDims().ndims() == 4 && srcDims[1] % 8 == 0) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nChw8c);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw8c);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nChw8c});
    }
}

void MKLDNNPadNode::selectOptimalPrimitiveDescriptor() {
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;

    bool checkPads = false;

    if (getParentEdgeAt(0)->getDims().ndims() == 4) {
        checkPads = true;

        for (size_t i = 0; i < padsBegin.size(); i++)
            if (padsBegin[i] != 0) {
                checkPads = false;
                break;
            }

        if (padsEnd[0] != 0 || padsEnd[1] != 0)
            checkPads = false;
    }

    if ((padMode == CONSTANT) && checkPads)
        canUseOptimalImpl = true;

    auto parent = getParentEdgeAt(0)->getParent();
    if (parent->getSelectedPrimitiveDescriptor() != nullptr) {
        for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
            auto inConfsDesc = supportedPrimitiveDescriptors[i].getConfig().inConfs[0].desc;
            auto parentOutConfsDesc = parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc;

            if ((inConfsDesc.getDims().size() == 4) &&
                (parentOutConfsDesc.getLayout() == InferenceEngine::Layout::BLOCKED) &&
                (inConfsDesc.getLayout() == InferenceEngine::Layout::BLOCKED) && canUseOptimalImpl) {
                selectPrimitiveDescriptorByIndex(i);
                return;
            }
        }
    }

    selectPrimitiveDescriptorByIndex(0);
}

void MKLDNNPadNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    SizeVector srcDims = getParentEdgeAt(0)->getBlob()->getTensorDesc().getDims();
    SizeVector dstDims = getChildEdgeAt(0)->getBlob()->getTensorDesc().getDims();

    params.srcStrides = getParentEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getStrides();
    params.dstStrides = getChildEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getStrides();
    params.padDims.resize(padsBegin.size());
    for (size_t i = 0; i < srcDims.size(); i++) {
        params.srcODms.push_back(srcDims[i] + padsBegin[i]);
        params.padPointsNum += padsBegin[i] + padsEnd[i];
        params.padDims[i] = padsBegin[i] + padsEnd[i];
    }
}

void MKLDNNPadNode::execute(mkldnn::stream strm) {
    const float *srcData = getParentEdgeAt(0)->getBlob()->cbuffer().as<const float *>() +
            getParentEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getOffsetPadding();
    float* dstData = getChildEdgeAt(0)->getBlob()->cbuffer().as<float *>() +
            getChildEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getOffsetPadding();

    auto srcDims = getParentEdgeAt(0)->getDims().ToSizeVector();
    auto dstDims = getChildEdgeAt(0)->getDims().ToSizeVector();

    switch (padMode) {
        case CONSTANT:
            if (canUseOptimalImpl) {
                if (getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getLayout() == InferenceEngine::Layout::BLOCKED)
                    padConstantOptimalImplForBlocked(srcData, dstData, srcDims);
                else
                    padConstantOptimalImpl(srcData, dstData, srcDims);
            } else {
                padConstant(srcData, dstData, srcDims, dstDims);
            }
            break;
        case EDGE:
            padEdge(srcData, dstData, srcDims, dstDims);
            break;
        case REFLECT:
            padReflect(srcData, dstData, srcDims, dstDims);
            break;
        case SYMMETRIC:
            padSymmetric(srcData, dstData, srcDims, dstDims);
            break;
    }
}

inline size_t parallel_init(size_t start, size_t size, std::vector<size_t> &counters, std::vector<size_t> &dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

inline void parallel_step(size_t size, std::vector<size_t> &counters, std::vector<size_t> &dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = (counters[j] + 1) % dims[j];
        if (counters[j] != 0)
            return;
    }
}

void MKLDNNPadNode::padConstant(const float *srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = srcDims[dimsSize_1];
    size_t workAmountSrc = params.srcStrides[0] * srcDims[0] / srcDims[dimsSize_1];

    int offset = 0;
    for (size_t i = 0; i < params.srcStrides.size(); ++i)
        offset += padsBegin[i] * params.dstStrides[i];
    std::fill_n(dstData, offset, padValue);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountSrc, nthr, ithr, start, end);
        SizeVector countersNext(dimsSize_1, 0);

        parallel_init(start, dimsSize_1, counters, srcDims);
        parallel_init(start, dimsSize_1, countersNext, srcDims);
        parallel_step(dimsSize_1, countersNext, srcDims);
        int srcIdx = 0;
        int dstIdx = padsBegin[dimsSize_1];
        int dstIdxNext = padsBegin[dimsSize_1];
        for (size_t i = 0; i < dimsSize_1; ++i) {
            srcIdx += counters[i] * params.srcStrides[i];
            dstIdx += (padsBegin[i] + counters[i]) * params.dstStrides[i];
            dstIdxNext += (padsBegin[i] + countersNext[i]) * params.dstStrides[i];
        }
        if (dstIdxNext <= dstIdx) dstIdxNext = params.dstStrides[0] * dstDims[0];

        for (size_t iwork = start; iwork < end; ++iwork, srcIdx += inputSV) {
            cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], sizeof(float) * inputSV);
            std::fill_n(&dstData[dstIdx + inputSV], dstIdxNext - dstIdx - inputSV, padValue);

            for (int j = dimsSize_1 - 1; j >= 0; j--) {
                counters[j] = (counters[j] + 1) % srcDims[j];
                if (counters[j] != 0) {
                    dstIdx += params.dstStrides[j];
                    break;
                } else {
                    dstIdx = padsBegin[dimsSize_1];
                    for (size_t i = 0; i < dimsSize_1; ++i)
                        dstIdx += (padsBegin[i] + counters[i]) * params.dstStrides[i];
                }
            }

            for (int j = dimsSize_1 - 1; j >= 0; j--) {
                countersNext[j] = (countersNext[j] + 1) % srcDims[j];
                if (countersNext[j] != 0) {
                    dstIdxNext += params.dstStrides[j];
                    break;
                } else {
                    dstIdxNext = padsBegin[dimsSize_1];
                    for (size_t i = 0; i < dimsSize_1; ++i)
                        dstIdxNext += (padsBegin[i] + countersNext[i]) * params.dstStrides[i];
                }
            }
            if (dstIdxNext <= dstIdx) dstIdxNext = params.dstStrides[0] * dstDims[0];
        }
    });
}

void MKLDNNPadNode::padConstantOptimalImpl(const float *srcData, float* dstData, SizeVector srcDims) {
    size_t outer_size = srcDims[0] * srcDims[1];
    uint8_t sizeData = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();

    parallel_for2d(srcDims[2], outer_size, [&](size_t i, size_t j) {
        auto dstPtr = dstData + i * params.dstStrides[2] + j * params.dstStrides[1];
        auto srcPtr = srcData + i * params.srcStrides[2] + j * params.srcStrides[1];

        cpu_memcpy(dstPtr, srcPtr, srcDims[3] * sizeData);

        if (padsEnd[3] != 0)
            std::fill_n(dstPtr + params.srcStrides[2], padsEnd[3], padValue);
    });

    if (padsEnd[2] != 0)
        parallel_for(outer_size, [&](size_t i) {
            auto dstPtr = dstData + srcDims[2] * params.dstStrides[2] + i * params.dstStrides[1];

            std::fill_n(dstPtr, padsEnd[2] * params.dstStrides[2], padValue);
        });
}

void MKLDNNPadNode::padConstantOptimalImplForBlocked(const float *srcData, float* dstData, SizeVector srcDims) {
    uint8_t sizeData = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();
    size_t blocksCount = srcDims[1] / 8;

    parallel_for3d(srcDims[2], blocksCount, srcDims[0], [&](size_t i, size_t j, size_t k) {
        auto srcPtr = srcData + i * params.srcStrides[2] + j * params.srcStrides[1] + k * params.srcStrides[0];
        auto dstPtr = dstData + i * params.dstStrides[2] + j * params.dstStrides[1] + k * params.dstStrides[0];

        cpu_memcpy(dstPtr, srcPtr, params.srcStrides[2] * sizeData);

        if (padsEnd[3] != 0)
            std::fill_n(dstPtr + params.srcStrides[2], 8 * padsEnd[3], padValue);
    });

    if (padsEnd[2] != 0)
        parallel_for2d(blocksCount, srcDims[0], [&](size_t i, size_t j) {
            auto dstPtr = dstData + srcDims[2] * params.dstStrides[2] + i * params.dstStrides[1] + j * params.dstStrides[0];

            std::fill_n(dstPtr, padsEnd[2] * params.dstStrides[2], padValue);
        });
}

void MKLDNNPadNode::padEdge(const float *srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = dstDims[dimsSize_1];
    size_t workAmountDst = params.dstStrides[0] * dstDims[0] / dstDims[dimsSize_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountDst, nthr, ithr, start, end);

        parallel_init(start, dimsSize_1, counters, dstDims);
        int dst_idx = 0;
        for (size_t i = 0; i < dimsSize_1; ++i)
            dst_idx += counters[i] * params.dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dst_idx += inputSV) {
            int src_idx = 0;
            for (size_t i = 0; i < dimsSize_1; ++i) {
                int idx = (counters[i] < padsBegin[i]) ? 0 :
                          ((counters[i] >= params.srcODms[i]) ? (srcDims[i] - 1) : (counters[i] - padsBegin[i]));
                src_idx += idx * params.srcStrides[i];
            }

            std::fill_n(&dstData[dst_idx], padsBegin[dimsSize_1], srcData[src_idx]);
            cpu_memcpy(&dstData[dst_idx + padsBegin[dimsSize_1]], &srcData[src_idx],
                       sizeof(float) * srcDims[dimsSize_1]);
            std::fill_n(&dstData[dst_idx + params.srcODms[dimsSize_1]], dstDims[dimsSize_1] - params.srcODms[dimsSize_1],
                        srcData[src_idx + srcDims[dimsSize_1] - 1]);

            parallel_step(dimsSize_1, counters, dstDims);
        }
    });
}

void MKLDNNPadNode::padReflect(const float *srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    SizeVector src_2;
    for (size_t i = 0; i < srcDims.size(); i++)
        src_2.push_back(srcDims[i] + params.srcODms[i] - 2);

    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = dstDims[dimsSize_1];
    size_t workAmountDst = params.dstStrides[0] * dstDims[0] / dstDims[dimsSize_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountDst, nthr, ithr, start, end);

        parallel_init(start, dimsSize_1, counters, dstDims);
        int dst_idx = 0;
        for (size_t i = 0; i < dimsSize_1; ++i)
            dst_idx += counters[i] * params.dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dst_idx += inputSV) {
            int src_idx = 0;
            for (size_t i = 0; i < dimsSize_1; ++i) {
                int idx = (counters[i] < padsBegin[i]) ? (padsBegin[i] - counters[i]) :
                          ((counters[i] >= params.srcODms[i]) ? (src_2[i] - counters[i]) : (counters[i] - padsBegin[i]));
                src_idx += idx * params.srcStrides[i];
            }

            for (size_t i = 0; i < padsBegin[dimsSize_1]; ++i) {
                dstData[dst_idx + i] = srcData[src_idx + padsBegin[dimsSize_1] - i];
            }
            cpu_memcpy(&dstData[dst_idx + padsBegin[dimsSize_1]], &srcData[src_idx], sizeof(float) * srcDims[dimsSize_1]);
            for (size_t i = params.srcODms[dimsSize_1]; i < dstDims[dimsSize_1]; ++i) {
                dstData[dst_idx + i] = srcData[src_idx + src_2[dimsSize_1] - i];
            }

            parallel_step(dimsSize_1, counters, dstDims);
        }
    });
}

void MKLDNNPadNode::padSymmetric(const float *srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    SizeVector src_2;
    for (size_t i = 0; i < srcDims.size(); i++)
        src_2.push_back(srcDims[i] + params.srcODms[i] - 1);

    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = dstDims[dimsSize_1];
    size_t workAmountDst = params.dstStrides[0] * dstDims[0] / dstDims[dimsSize_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountDst, nthr, ithr, start, end);

        parallel_init(start, dimsSize_1, counters, dstDims);
        int dst_idx = 0;
        for (size_t i = 0; i < dimsSize_1; ++i)
            dst_idx += counters[i] * params.dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dst_idx += inputSV) {
            int src_idx = 0;
            for (size_t i = 0; i < dimsSize_1; ++i) {
                int idx = (counters[i] < padsBegin[i]) ? (padsBegin[i] - 1 - counters[i]) :
                          ((counters[i] >= params.srcODms[i]) ? (src_2[i] - counters[i]) : (counters[i] - padsBegin[i]));
                src_idx += idx * params.srcStrides[i];
            }

            for (size_t i = 0; i < padsBegin[dimsSize_1]; ++i)
                dstData[dst_idx + i] = srcData[src_idx + padsBegin[dimsSize_1] -1 - i];

            cpu_memcpy(&dstData[dst_idx + padsBegin[dimsSize_1]], &srcData[src_idx],
                       sizeof(float) * srcDims[dimsSize_1]);

            for (size_t i = params.srcODms[dimsSize_1]; i < dstDims[dimsSize_1]; ++i)
                dstData[dst_idx + i] = srcData[src_idx + src_2[dimsSize_1] - i];

            parallel_step(dimsSize_1, counters, dstDims);
        }
    });
}

bool MKLDNNPadNode::created() const {
    return getType() == Pad;
}
REG_MKLDNN_PRIM_FOR(MKLDNNPadNode, Pad);
