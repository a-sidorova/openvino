// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {
// ===================================Add=========================================================//
// These  inputs are needed to test static Loop optimizations (emit the whole tile, body with increments, set WA etc)
std::vector<ov::test::InputShape> inShapesStatic1{{{}, {{128, 256, 512}}}};
std::vector<ov::test::InputShape> inShapesStatic2{{{}, {{128, 256, 512}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Add,
                         ::testing::Combine(
                             ::testing::ValuesIn(inShapesStatic1),
                             ::testing::ValuesIn(inShapesStatic2),
                             ::testing::ValuesIn({ov::element::f32}),
                             ::testing::Values(1), // Add
                             ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                             ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         Add::getTestCaseName);


} // namespace
} // namespace snippets
} // namespace test
} // namespace ov