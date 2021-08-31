// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     BroadcastMatMul transformation detects MatMul operations
 *     and broadcasts one input to another. This transformation is
 *     required because of oneDNN 1.6 restrictions which allows only 3D shapes for MatMul input.
 *     Used together with ReshapeMatMul.
 *     ReshapeMatMul and BroadcastMatMul should be replaced with single Unsqueeze transformation
 *     After migration to oneDNN 2.3
 */

namespace MKLDNNPlugin {

class BroadcastMatMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BroadcastMatMul();
};

}  // namespace MKLDNNPlugin
