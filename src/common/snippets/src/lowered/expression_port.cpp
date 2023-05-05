// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_port.hpp"

#include "snippets/utils.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

ExpressionPort::ExpressionPort(const std::weak_ptr<Expression>& expr, Type type, size_t port,
                               const std::vector<size_t>& tensor, const std::vector<size_t>& layout, const std::vector<size_t>& subtensor)
        : m_expr(expr), m_type(type), m_port_index(port), m_port_desc(std::make_shared<PortDescriptor>(tensor, subtensor, layout)) {}

ExpressionPort::ExpressionPort(const std::weak_ptr<Expression>& expr, Type type, size_t port, const PortDescriptorPtr& port_desc)
        : m_expr(expr), m_type(type), m_port_index(port) {
    PortDescriptorPtr local_port_desc = port_desc;
    if (!local_port_desc) {
        if (type == Type::Input) {
            local_port_desc = PortManager::get_port_descriptor_ptr(expr.lock()->get_node()->input(port));
        } else if (type == Type::Output) {
            local_port_desc = PortManager::get_port_descriptor_ptr(expr.lock()->get_node()->output(port));
        } else {
            OPENVINO_THROW("ExpressionPort supports only Input and Output type!");
        }
    }

    m_port_desc = local_port_desc;
}

std::shared_ptr<Expression> ExpressionPort::get_expr_ptr() const {
    auto shared = m_expr.lock();
    OPENVINO_ASSERT(shared != nullptr, "Failed attempt to get shared pointer of source expression: nullptr");
    return shared;
}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
