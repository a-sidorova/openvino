// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_reshape_elimination_for_matmul.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeReshapeEliminationForMatmul, "TransposeReshapeEliminationForMatmul", 0);

ngraph::pass::TransposeReshapeEliminationForMatmul::TransposeReshapeEliminationForMatmul() {
    MATCHER_SCOPE(TransposeReshapeEliminationForMatmul);
    auto input_1_pattern = ngraph::pattern::any_input();
    auto input_2_pattern = ngraph::pattern::any_input();

    auto const_transpose_before_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto transpose_before_pattern = ngraph::pattern::wrap_type<opset1::Transpose>({input_2_pattern, const_transpose_before_pattern});

    auto const_reshape_before_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto reshape_before_pattern = ngraph::pattern::wrap_type<opset1::Reshape>({transpose_before_pattern, const_reshape_before_pattern});

    auto matmul_pattern = ngraph::pattern::wrap_type<opset1::MatMul>({input_1_pattern, reshape_before_pattern});

    auto const_reshape_after_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto reshape_after_pattern = ngraph::pattern::wrap_type<opset1::Reshape>({matmul_pattern, const_reshape_after_pattern});

    auto const_transpose_after_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto transpose_after_pattern = ngraph::pattern::wrap_type<opset1::Transpose>({reshape_after_pattern, const_transpose_after_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input_1 = pattern_value_map.at(input_1_pattern);
        const auto& input_2 = pattern_value_map.at(input_2_pattern);
        auto transpose_before = std::dynamic_pointer_cast<opset1::Transpose>(pattern_value_map.at(transpose_before_pattern).get_node_shared_ptr());
        auto reshape_before = std::dynamic_pointer_cast<opset1::Reshape>(pattern_value_map.at(reshape_before_pattern).get_node_shared_ptr());
        auto matmul = std::dynamic_pointer_cast<opset1::MatMul>(pattern_value_map.at(matmul_pattern).get_node_shared_ptr());
        auto reshape_after = std::dynamic_pointer_cast<opset1::Reshape>(pattern_value_map.at(reshape_after_pattern).get_node_shared_ptr());
        auto transpose_after = std::dynamic_pointer_cast<opset1::Transpose>(pattern_value_map.at(transpose_after_pattern).get_node_shared_ptr());

        const auto new_matmul = std::make_shared<opset1::MatMul>(input_1, input_2);
        new_matmul->set_friendly_name(matmul->get_friendly_name());
        copy_runtime_info({transpose_before, reshape_before, matmul, reshape_after, transpose_after}, new_matmul);
        replace_node(transpose_after, new_matmul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_after_pattern, matcher_name);
    this->register_matcher(m, callback);
}
