// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

#include "snippets/op/subgraph.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

class CommonOptimizations : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    CommonOptimizations();

private:
    // Move up Constants which aren't scalars from body to Subgraph and replace them with Parameters inside body
    void ExtractConstants(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph);
    // Move up unsupported Transposes after Parameters from body
    void ExtractUnsupportedTransposes(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
