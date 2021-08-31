// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_matmul.hpp"

#include "ngraph/op/matmul.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

#include <transformations/utils/utils.hpp>

#include <algorithm>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ReshapeMatMul, "ReshapeMatMul", 0);

MKLDNNPlugin::ReshapeMatMul::ReshapeMatMul() {
    ngraph::OutputVector twoInputs = {
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())
    };

    auto fcTwoInputs = ngraph::pattern::wrap_type<ngraph::op::MatMul>(twoInputs, ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::op::MatMul> (m.get_match_root());

        if (!matmul || transformation_callback(matmul))
            return false;

        auto input0_shape = matmul->input_value(0).get_shape();
        auto input1_shape = matmul->input_value(1).get_shape();
        auto output_shape = matmul->get_shape();

        if (input0_shape.size() <= 3 && input1_shape.size() <= 3)
            return false;

        ngraph::NodeVector new_ops;
        std::shared_ptr<ngraph::Node> newReshapeInput0;
        std::shared_ptr<ngraph::Node> newReshapeInput1;

        auto getReshape = [](const ngraph::Output<ngraph::Node>& reshapeInput) {
            const auto& inputShape = reshapeInput.get_shape();
            std::vector<int64_t> reshape_shape{-1, static_cast<int64_t>(inputShape.rbegin()[1]), static_cast<int64_t>(inputShape.rbegin()[0])};
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(reshapeInput,
                                                                     ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, reshape_shape),
                                                                     true);
            reshape->set_friendly_name(reshapeInput.get_node()->get_friendly_name() + "/Reshape");
            return reshape;
        };

        if (input0_shape.size() > 3) {
            newReshapeInput0 = getReshape(matmul->input_value(0));
            new_ops.push_back(newReshapeInput0);
        }

        if (input1_shape.size() > 3) {
            newReshapeInput1 = getReshape(matmul->input_value(1));
            new_ops.push_back(newReshapeInput1);
        }

        std::shared_ptr<ngraph::Node> matmul_new;

        matmul_new = std::make_shared<ngraph::op::MatMul>(newReshapeInput0 ? newReshapeInput0 : matmul->input_value(0),
                                                          newReshapeInput1 ? newReshapeInput1 : matmul->input_value(1),
                                                          matmul->get_transpose_a(),
                                                          matmul->get_transpose_b());
        new_ops.push_back(matmul_new);

        auto reshape_output = ngraph::op::util::reshapeTo(matmul_new, output_shape);
        new_ops.push_back(reshape_output);
        reshape_output->set_friendly_name(matmul->get_friendly_name());
        matmul_new->set_friendly_name(matmul->get_friendly_name() + "/MM");
        ngraph::copy_runtime_info(matmul, new_ops);
        ngraph::replace_node(matmul, reshape_output);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fcTwoInputs, "ReshapeMatMul");
    this->register_matcher(m, callback);
}
