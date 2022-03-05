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
 * @interface InsertConvertAfter
 * @brief Inserts explicit convert node after Load, BroadcastLoad and Scalars to align precision
 * @ingroup snippets
 */
class InsertConvertAfter: public ngraph::pass::MatcherPass {
public:
    InsertConvertAfter(const ov::element::TypeVector& supported_exec_types);
};

/**
 * @interface InsertConvertBeforeStore
 * @brief Inserts explicit convert node before to align precision
 * @ingroup snippets
 */
class InsertConvertBeforeStore: public ngraph::pass::MatcherPass {
public:
    InsertConvertBeforeStore(const ov::element::TypeVector& supported_exec_types);
};

/**
 * @interface PrecisionPropagation
 * @brief Propagate precision inside body to align precision between nodes. Should be called after all Convert insertions
 * @ingroup snippets
 */
class PrecisionPropagation: public ngraph::pass::FunctionPass {
public:
    PrecisionPropagation(const ov::element::Type default_type = ov::element::f32);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
private:
    ov::element::Type default_type;
};

/**
 * @interface InsertConvert
 * @brief Inserts explicit convert to align precisions inside subgraph
 * @ingroup snippets
 */
class InsertConvert: public ngraph::pass::GraphRewrite {
public:
    InsertConvert(const ov::element::TypeVector& supported_exec_types) {
        add_matcher<InsertConvertAfter>(supported_exec_types);
        add_matcher<InsertConvertBeforeStore>(supported_exec_types);
    }
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
