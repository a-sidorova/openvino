// Copyright (C) 2023 Intel Corporation
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

    static bool canBeParallelOptimized(const std::shared_ptr<const ov::Node>& node);

private:
    // Move up Constants which aren't scalars from body to Subgraph and replace them with Parameters inside body
    static void ExtractConstants(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph);
    // Move up unsupported Transposes after Parameters from body
    static bool ExtractUnsupportedTransposes(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph);
    // Insert Reshape nodes around Subgraph to increase work amount for parallelism
    static bool SplitDimensions(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
