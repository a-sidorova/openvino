// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_quantize_precision_propagation_function.hpp"

#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
std::shared_ptr<ngraph::op::FakeQuantize> makeFakeQuantize(
    const Output<Node>& parent,
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint) {
    auto generate = [](const ov::element::Type precision,
        const ngraph::Shape& shape,
        const float initialValue,
        const std::string& name) {
            const auto size = ngraph::shape_size(shape);
            std::vector<float> values(size);
            for (auto i = 0; i < size; ++i) {
                values[i] = static_cast<float>(initialValue + i);
            }
            auto constant = std::make_shared<ngraph::opset1::Constant>(precision, shape, values);
            constant->set_friendly_name(name);
            return constant;
    };

    const auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(
        parent,
        generate(inputType, fakeQuantizeShapes[0], zeroPoint, "inputLow"),
        generate(inputType, fakeQuantizeShapes[1], 20.f, "inputHigh"),
        generate(inputType, fakeQuantizeShapes[2], zeroPoint, "outputLow"),
        generate(inputType, fakeQuantizeShapes[3], 20.f, "outputHigh"),
        256ul);
    fakeQuantize->set_friendly_name("fakeQuantize");

    return fakeQuantize;
}
} // namespace

std::shared_ptr<ov::Model> FakeQuantizePrecisionPropagationFunction::get(
    const ngraph::Shape& inputShape,
    const element::Type inputType) {
    const auto parameter1 = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter1->set_friendly_name("parameter1");

    const auto parameter2 = std::make_shared<ngraph::opset1::Parameter>(inputType, ov::PartialShape{1, 1, 1, 16});
    parameter2->set_friendly_name("parameter2");

    std::shared_ptr<Node> parent = std::make_shared<ngraph::opset1::MaxPool>(
        parameter1,
        Strides{ 1, 1 }, // strides
        Shape{ 0, 0 },   // pads_begin
        Shape{ 0, 0 },   // pads_end
        Shape{ 1, 1 });  // kernel
    parent->set_friendly_name("maxPool");

    parent = makeFakeQuantize(
        parent,
        inputShape,
        inputType,
        { {}, {}, {}, {} }, // fakeQuantizeShapes,
        0.f);
    parent->set_friendly_name("fakeQuantize");

    parent = std::make_shared<ngraph::opset1::Add>(parent, parameter2);
    parent->set_friendly_name("add");

    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        ParameterVector{ parameter1, parameter2 },
        "FakeQuantizeFunction");

    function->validate_nodes_and_infer_types();

    return function;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
