// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::PartialShape>> inputShapes = {
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 1, 1, 128}, {1, 128, 16, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 16, 1, 1}, {1, 128, 16, 64}},
        {{2, 68, 6, 92}, {2, 68, 6, 92}, {1, 1, 68, 68}, {2, 68, 6, 92}},
        {{1, 58, 16, 34}, {1, 58, 16, 34}, {1, 1, 1, 58}, {1, 58, 16, 34}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHA,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShapes),
                             ::testing::ValuesIn({false, true}),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapeSelect = {
        // without broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 94, 12, 54}, {1, 94, 12, 54}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 94, 12, 54}},
        // with broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 1, 1}, {1, 12, 1, 1}, {1, 128, 12, 64}},
        {{2, 52, 6, 102}, {2, 52, 6, 102}, {1, 6, 52, 52}, {1, 6, 1, 1}, {1, 6, 1, 1}, {2, 52, 6, 102}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHASelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeSelect),
                                 ::testing::Values(false),  // Need to support True for graph builder in tests
                                 ::testing::Values(2), // Less + MHA
                                 ::testing::Values(2),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);


const std::vector<std::vector<ov::PartialShape>> inputShapesWOTranspose = {
        {{1, 12, 197, 64}, {1, 12, 64, 197}, {1, 12, 197, 64}},
        {{12, 197, 64}, {12, 64, 197}, {12, 197, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTranspose, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::ValuesIn({false}),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHAWOTranspose::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeOnInputs, MHAWOTransposeOnInputs,
                         ::testing::Combine(
                                 ::testing::Values(inputShapesWOTranspose[0]), // Only 4D
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::ValuesIn({false}),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHAWOTranspose::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapesWOTransposeMatMul0TransposedB = {
        {{1, 12, 197, 64}, {1, 12, 197, 64}, {1, 12, 197, 64}},
        {{12, 197, 64}, {12, 197, 64}, {12, 197, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeMatMul0TransposedB, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTransposeMatMul0TransposedB),
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::ValuesIn({true}),
                                 ::testing::Values(2), // Extracted Transpose + MHA
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHAWOTranspose::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeOnInputsMatMul0TransposedB, MHAWOTransposeOnInputs,
                         ::testing::Combine(
                                 ::testing::Values(inputShapesWOTransposeMatMul0TransposedB[0]), // Only 4D
                                 ::testing::ValuesIn({true, false}),
                                 ::testing::ValuesIn({true}),
                                 ::testing::Values(2), // Extracted Transpose + MHA
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHAWOTranspose::getTestCaseName);


} // namespace
} // namespace snippets
} // namespace test
} // namespace ov