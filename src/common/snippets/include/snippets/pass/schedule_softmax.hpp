// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface ScheduleSoftmax
 * @brief The pass updates port descriptors for Softmax to show by which axes there is reducing
 * @ingroup snippets
 */
class ScheduleSoftmax: public ngraph::pass::MatcherPass {
public:
    ScheduleSoftmax();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
