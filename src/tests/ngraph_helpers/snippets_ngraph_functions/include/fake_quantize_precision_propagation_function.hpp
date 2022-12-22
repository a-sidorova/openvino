// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"


namespace ov {
namespace test {
namespace snippets {

class FakeQuantizePrecisionPropagationFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ngraph::Shape& inputShape,
        const element::Type inputType);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
