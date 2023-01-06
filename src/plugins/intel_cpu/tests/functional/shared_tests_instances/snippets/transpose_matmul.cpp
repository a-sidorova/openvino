// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose_matmul.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_system_conf.h"

namespace ov {
namespace test {
namespace snippets {


namespace {
static inline std::vector<std::vector<element::Type>> precisions() {
    std::vector<std::vector<element::Type>> prc = {
            {element::f32, element::f32},
            {element::i8, element::i8},
            {element::u8, element::i8}
    };
    if (InferenceEngine::with_cpu_x86_bfloat16() || InferenceEngine::with_cpu_x86_avx512_core_amx_bf16()) {
        prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
    }
    return prc;
}
namespace transpose_zero_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{1, 49, 2, 23}, {2, 2, 23, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(0), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

// TODO: FuseTransposeToBrgemm supports fusing only if Transpose is before Parameter in cases when Transpose is on input
// INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ, TransposeMatMulFQ,
//                          ::testing::Combine(
//                                  ::testing::ValuesIn(transpose_input_shapes),
//                                  ::testing::Values(0), // Transpose on 0th Matmul input
//                                  ::testing::Values(ov::element::i8),
//                                  ::testing::Values(1), // MatMul
//                                  ::testing::Values(1), // Tokenized MatMul + FusedTranspose
//                                  ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                          TransposeMatMulFQ::getTestCaseName);
} // namespace transpose_zero_input

namespace transpose_first_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 49, 13}, {1, 13, 3, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ, TransposeMatMulFQ,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::Values(std::vector<element::Type>{ov::element::f32}),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMulFQ::getTestCaseName);
} // namespace transpose_first_input

namespace transpose_output {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 49, 13}, {1, 2, 13, 39}}
};
// TODO: Propagate shape through Brgemm with Transpose down
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(2), // Transpose on Matmul output
                                 ::testing::Values(std::vector<element::Type>{ov::element::f32, ov::element::f32}),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

// TODO: Propagate shape through Brgemm with Transpose down
// INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ, TransposeMatMulFQ,
//                          ::testing::Combine(
//                                  ::testing::ValuesIn(transpose_input_shapes),
//                                  ::testing::Values(2), // Transpose on Matmul output
//                                  ::testing::Values(ov::element::i8),
//                                  ::testing::Values(1), // MatMul
//                                  ::testing::Values(1), // Tokenized MatMul + FusedTranspose
//                                  ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                          TransposeMatMulFQ::getTestCaseName);
} // namespace transpose_output

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov