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
 * @interface ReplaceLoadsWithScalarLoads
 * @brief Set count `1` for Load to represent as ScalarLoad
 * The pass is used to change element count to loading to "1" to load scalar value
 * Used for tail generation
 * @ingroup snippets
 */
class ReplaceLoadsWithScalarLoads: public ngraph::pass::MatcherPass {
public:
    ReplaceLoadsWithScalarLoads();
};

/**
 * @interface ReplaceStoresWithScalarStores
 * @brief Set count `1` for Store to represent as ScalarStore
 * The pass is used to change element count to stroring to "1" to store scalar valuw
 * Used for tail generation
 * @ingroup snippets
 */
class ReplaceStoresWithScalarStores: public ngraph::pass::MatcherPass {
public:
    ReplaceStoresWithScalarStores();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
