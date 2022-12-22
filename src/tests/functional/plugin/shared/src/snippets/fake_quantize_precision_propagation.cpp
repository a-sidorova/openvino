// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/fake_quantize_precision_propagation.hpp"

#include "common_test_utils/common_utils.hpp"
#include "fake_quantize_precision_propagation_function.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string FakeQuantizePrecisionPropagation::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj) {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, withMul, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i)
        result << "IS[" << i << "]=" << CommonTestUtils::partialShape2str({inputShapes[i]}) << "_";
    result << "Mul=" << withMul << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void FakeQuantizePrecisionPropagation::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    std::tie(inputShapes, withMul, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    function = ov::test::snippets::FakeQuantizePrecisionPropagationFunction::get({1, 3, 16, 16}, ov::element::f32);
    //ngraph::pass::VisualizeTree("svg/test.actual.svg").run_on_model(function);
}

//void FakeQuantizePrecisionPropagation::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
//    inputs.clear();
//    auto model_inputs = function->inputs();
//    for (auto& model_input : model_inputs) {
//        const auto node_input = model_input.get_node_shared_ptr();
//        const auto name = node_input->get_friendly_name();
//        ov::Tensor tensor;
//        int seed = 0;
//        if (name.find("less") != std::string::npos) {
//            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), model_input.get_shape(), 5 + seed, -2, 10, seed);
//            seed++;
//        } else {
//            tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(model_input.get_element_type(), model_input.get_shape(), 1.0f, 0.5f);
//        }
//        inputs.insert({node_input, tensor});
//    }
//}

TEST_P(FakeQuantizePrecisionPropagation, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
