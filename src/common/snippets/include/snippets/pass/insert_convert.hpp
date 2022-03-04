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
 * @interface InsertConvertAfterLoad
 * @brief Inserts explicit convert node after Load and BroadcastLoad to align precision
 * @ingroup snippets
 */
class InsertConvertAfterLoad: public ngraph::pass::MatcherPass {
public:
    InsertConvertAfterLoad(const ov::element::TypeVector& supported_exec_types);
};

/**
 * @interface InsertConvertBeforeStore
 * @brief Inserts explicit convert node after Load to align precision
 * @ingroup snippets
 */
class InsertConvertBeforeStore: public ngraph::pass::MatcherPass {
public:
    InsertConvertBeforeStore();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
