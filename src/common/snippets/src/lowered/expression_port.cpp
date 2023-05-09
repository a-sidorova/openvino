// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_port.hpp"

#include "snippets/utils.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

ExpressionPort::ExpressionPort(const std::shared_ptr<Expression>& expr, Type type, size_t port)
        : m_expr(expr), m_type(type), m_port_index(port) {}

PortDescriptorPtr ExpressionPort::get_port_descriptor() const {
    const auto& descs = m_type == Type::Input ? m_expr->m_input_port_descriptors
                                              : m_expr->m_output_port_descriptors;
    OPENVINO_ASSERT(m_port_index < descs.size(), "Incorrect index of port");
    return descs[m_port_index];
}

const std::shared_ptr<Tensor>& ExpressionPort::get_tensor_ptr() const {
    const auto& tensors = m_type == Type::Input ? m_expr->m_input_tensors
                                                : m_expr->m_output_tensors;
    OPENVINO_ASSERT(m_port_index < tensors.size(), "Incorrect index of port");
    return tensors[m_port_index];
}

std::vector<size_t> ExpressionPort::get_tensor() const {
    return get_port_descriptor()->get_tensor();
}
std::vector<size_t> ExpressionPort::get_layout() const {
    return get_port_descriptor()->get_layout();
}
std::vector<size_t> ExpressionPort::get_subtensor() const {
    return get_port_descriptor()->get_subtensor();
}

void ExpressionPort::set_tensor(const std::vector<size_t>& tensor) {
    get_port_descriptor()->set_tensor(tensor);
}
void ExpressionPort::set_layout(const std::vector<size_t>& layout) {
    get_port_descriptor()->set_layout(layout);
}
void ExpressionPort::set_subtensor(const std::vector<size_t>& subtensor) {
    get_port_descriptor()->set_subtensor(subtensor);
}

bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    if (&lhs == &rhs)
        return true;
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect ExpressionPort comparison");
    return lhs.get_index() == rhs.get_index() && lhs.get_expr() == rhs.get_expr();
}
bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    return !(lhs == rhs);
}
bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect ExpressionPort comparison");
    return (lhs.get_index() < rhs.get_index()) || (lhs.get_index() == rhs.get_index() && lhs.get_expr() < rhs.get_expr());
}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
