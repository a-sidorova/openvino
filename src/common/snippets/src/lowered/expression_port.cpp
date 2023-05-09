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

const PortDescriptorPtr& ExpressionPort::get_descriptor_ptr() const {
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

std::set<ExpressionPort> ExpressionPort::get_connected_ports() const {
    if (ExpressionPort::m_type == Type::Input) {
        return { m_expr->m_input_tensors[m_port_index]->get_source() };
    }
    if (ExpressionPort::m_type == Type::Output) {
        return m_expr->m_output_tensors[m_port_index]->get_consumers();
    }
    OPENVINO_THROW("ExpressionPort supports only Input and Output types");
}

std::vector<size_t> ExpressionPort::get_shape() const {
    return get_descriptor_ptr()->get_shape();
}
std::vector<size_t> ExpressionPort::get_layout() const {
    return get_descriptor_ptr()->get_layout();
}
std::vector<size_t> ExpressionPort::get_subtensor() const {
    return get_descriptor_ptr()->get_subtensor();
}

void ExpressionPort::set_shape(const std::vector<size_t>& tensor) {
    get_descriptor_ptr()->set_shape(tensor);
}
void ExpressionPort::set_layout(const std::vector<size_t>& layout) {
    get_descriptor_ptr()->set_layout(layout);
}
void ExpressionPort::set_subtensor(const std::vector<size_t>& subtensor) {
    get_descriptor_ptr()->set_subtensor(subtensor);
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
