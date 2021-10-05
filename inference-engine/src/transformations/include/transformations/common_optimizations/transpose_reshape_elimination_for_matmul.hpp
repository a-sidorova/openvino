// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API TransposeReshapeEliminationForMatmul;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeReshapeEliminationForMatmul transformation eliminates Transpose and Reshape
 * which were created to align input and output dimension ranks before and after MatMul
 */
class ngraph::pass::TransposeReshapeEliminationForMatmul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeReshapeEliminationForMatmul();
};
