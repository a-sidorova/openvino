// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeMHASnippets
 * @brief The pass tokenizes MHA-pattern into Subgraph
 *        Pattern:           Transpose1
 *                               |
 *             Transpose0  Eltwise/Select
 *                     \     /
 *                     MatMul0
 *                        |
 *           Eltwise/Select/Reshape
 *                        |
 *                     Softmax
 *                        |
 *            Eltwise/Select/Reshape  Transpose2
 *                               \      /
 *                                MatMul1
 *                                  |
 *                  Eltwise/Select/Reshape/Transpose3
 *        Note: Transposes can be missed
 *       
 * @ingroup snippets
 */
class TokenizeMHASnippets: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TokenizeMHASnippets", "0");
    TokenizeMHASnippets(bool enable_transpose_tokenization = true);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
