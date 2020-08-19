template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::mish_compute_vector(
        const Vmm &vmm_src) {
    // Save src data on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // ln(1+exp(x))
    soft_relu_compute_vector(vmm_src);
    // tanh(ln(1+exp(x)))
    tanh_compute_vector(vmm_src);
    // x*tanh(ln(1+exp(x)))
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/range_add.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<float> positiveStart = { 1.0f, 1.2f };
const std::vector<float> positiveStop = { 5.0f, 5.2f };
const std::vector<float> positiveStep = { 1.0f, 0.1f };

const std::vector<float> negativeStart = { 1.0f, 1.2f };
const std::vector<float> negativeStop = { -5.0f, -5.2f };
const std::vector<float> negativeStep = { -1.0f, -0.1f };

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(BasicPositive, RangeAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(positiveStart),
                                ::testing::ValuesIn(positiveStop),
                                ::testing::ValuesIn(positiveStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BasicNegative, RangeAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(negativeStart),
                                ::testing::ValuesIn(negativeStop),
                                ::testing::ValuesIn(negativeStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeAddSubgraphTest::getTestCaseName);
}  // namespace
