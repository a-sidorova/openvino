// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_matmul.hpp"

#include "ngraph/op/matmul.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pattern/op/or.hpp>

#include <algorithm>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::BroadcastMatMul, "BroadcastMatMul", 0);

MKLDNNPlugin::BroadcastMatMul::BroadcastMatMul() {
    ngraph::OutputVector twoInputs = {
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())
    };

    auto matmulPattern = ngraph::pattern::wrap_type<ngraph::op::MatMul>(twoInputs, ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::op::MatMul> (m.get_match_root());

        if (!matmul || transformation_callback(matmul))
            return false;

        const auto& input0 = matmul->input_value(0);
        const auto& input1 = matmul->input_value(1);
        const auto& input0shape = input0.get_shape();
        const auto& input1shape = input1.get_shape();

        if (input0shape == input1shape)
            return false;

        if (input0shape.size() == 1 || input1shape.size() == 1)
            return false;

        if (input0shape.size() == input1shape.size() &&
            input0shape.size() > 2 &&
            // batch dimensions are equal
            std::equal(input0shape.begin(), input0shape.begin() + input0shape.size() - 2, input1shape.begin()))
            return false;

        std::shared_ptr<ngraph::Node> broadcastInput0;
        std::shared_ptr<ngraph::Node> broadcastInput1;
        ngraph::NodeVector new_ops;

        auto getBroadcast = [&](const ngraph::Output<ngraph::Node>& bcFrom, const ngraph::Output<ngraph::Node>& bcTo) {
            const auto& shapeFrom = bcFrom.get_shape();
            const auto& shapeTo = bcTo.get_shape();

            // only batch dims are broadcasted
            ngraph::PartialShape batchDimsFrom{ngraph::Shape{shapeFrom.begin(), shapeFrom.end() - 2}};
            const ngraph::PartialShape batchDimsTo{ngraph::Shape{shapeTo.begin(), shapeTo.end() - 2}};

            ngraph::PartialShape::broadcast_merge_into(batchDimsFrom, batchDimsTo, ngraph::op::AutoBroadcastType::NUMPY);

            ngraph::Shape target(batchDimsFrom.get_shape());
            target.insert(target.end(), shapeFrom.end() - 2, shapeFrom.end());

            auto broadcastInput = ngraph::op::util::broadcastTo(bcFrom, target);
            broadcastInput->set_friendly_name(bcFrom.get_node()->get_friendly_name() + "/BC");
            new_ops.push_back(broadcastInput);

            return broadcastInput;
        };

        if (input0shape.size() >= 2 && input1shape.size() > 2)
            broadcastInput0 = getBroadcast(input0, input1);
        if (input1shape.size() >= 2 && input0shape.size() > 2)
            broadcastInput1 = getBroadcast(input1, input0);

        std::shared_ptr<ngraph::Node> matmul_new;

        matmul_new = std::make_shared<ngraph::op::MatMul>(broadcastInput0 ? broadcastInput0 : input0,
                                                          broadcastInput1 ? broadcastInput1 : input1,
                                                          matmul->get_transpose_a(),
                                                          matmul->get_transpose_b());
        new_ops.push_back(matmul_new);
        matmul_new->set_friendly_name(matmul->get_friendly_name());

        ngraph::copy_runtime_info(matmul, new_ops);
        ngraph::replace_node(matmul, matmul_new);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmulPattern, "BroadcastMatMul");
    this->register_matcher(m, callback);
}
