// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/fake_quantize_precision_propagation.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

// TODO: FQ: U8 + I8
// TODO: FakeQuantizePrecisionPropagation => PrecisionPropagationConvertion
const std::vector<std::vector<ov::PartialShape>> inputShapeSelect = {
        // without broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 94, 12, 54}, {1, 94, 12, 54}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 94, 12, 54}},
        // with broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 1, 1}, {1, 12, 1, 1}, {1, 128, 12, 64}},
        {{2, 52, 6, 102}, {2, 52, 6, 102}, {1, 6, 52, 52}, {1, 6, 1, 1}, {1, 6, 1, 1}, {2, 52, 6, 102}}
};

//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_PrecisionPropagation_Convertion, FakeQuantizePrecisionPropagation,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(inputShapeSelect),
//                                 ::testing::Values(false),  // Need to support True for graph builder in tests
//                                 ::testing::Values(4),      // Less + MHA
//                                 ::testing::Values(1),
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                         FakeQuantizePrecisionPropagation::getTestCaseName);


} // namespace
} // namespace snippets
} // namespace test
} // namespace ov