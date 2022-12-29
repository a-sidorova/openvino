// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/buffer.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ngraph::snippets::op::Buffer);

auto normalize_rank(int32_t allocation_rank, const size_t shape_rank) -> int32_t {
    return allocation_rank < 0 ? allocation_rank + shape_rank : allocation_rank;
}

snippets::op::Buffer::Buffer(const Output<Node>& x, const int32_t allocation_rank)
    : Op({x}), m_allocation_rank(allocation_rank), m_is_single(false) {
    constructor_validate_and_infer_types();
}

snippets::op::Buffer::Buffer(const ov::Shape shape, const ov::element::Type element_type, const int32_t allocation_rank)
    : Op(), m_static_shape(shape), m_element_type(element_type), m_allocation_rank(allocation_rank), m_is_single(true) {
    constructor_validate_and_infer_types();
}

bool snippets::op::Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    visitor.on_attribute("allocation_rank", m_allocation_rank);
    if (m_is_single) {
        visitor.on_attribute("shape", m_static_shape);
        visitor.on_attribute("element_type", m_element_type);
    }
    return true;
}

std::shared_ptr<Node> snippets::op::Buffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (m_is_single) {
        return std::make_shared<Buffer>(m_static_shape, m_element_type, m_allocation_rank);
    }

    return std::make_shared<Buffer>(new_args.at(0), m_allocation_rank);
}

void snippets::op::Buffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    ov::PartialShape output_shape;
    ov::element::Type output_type;
    if (m_is_single) {
        output_shape = m_static_shape;
        output_type = m_element_type;
    } else {
        output_shape = get_input_partial_shape(0);
        output_type = get_input_element_type(0);
    }

    const auto shape_rank = output_shape.rank();
    if (shape_rank.is_static()) {
        const auto normalized_rank = normalize_rank(m_allocation_rank, shape_rank.get_length());
        NGRAPH_CHECK(normalized_rank >= 0 && normalized_rank <= shape_rank.get_length(),
                     "Buffer has incorrect allocation rank: " + std::to_string(m_allocation_rank));
    }

    set_output_type(0, output_type, output_shape);
}

size_t ngraph::snippets::op::Buffer::get_byte_size() const {
    const auto pshape = get_output_partial_shape(0);
    NGRAPH_CHECK(pshape.is_static(), "Buffer should have static shapes for memory allocation");
    const auto shape = pshape.get_shape();
    const auto normalized_rank = normalize_rank(m_allocation_rank, shape.size());
    return ngraph::shape_size(shape.rbegin(), shape.rbegin() + normalized_rank + 1) * get_element_type().size();
}
