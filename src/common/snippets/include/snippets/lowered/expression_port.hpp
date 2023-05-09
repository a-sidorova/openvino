// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "snippets/port_descriptor.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

class Tensor;
class Expression;
class ExpressionPort {
public:
    enum Type {
        Input,
        Output
    };

    ExpressionPort() = default;
    explicit ExpressionPort(const std::shared_ptr<Expression>& expr, Type type, size_t port);

    const std::shared_ptr<Expression>& get_expr() const { return m_expr; }
    Type get_type() const { return m_type; }
    size_t get_index() const { return m_port_index; }

    std::vector<size_t> get_shape() const;
    std::vector<size_t> get_layout() const;
    std::vector<size_t> get_subtensor() const;
    const PortDescriptorPtr& get_descriptor_ptr() const;
    const std::shared_ptr<Tensor>& get_tensor_ptr() const;
    // Returns connected ports to the current:
    //  - Input port returns one source (parent) port
    //  - Output port returns all consumer ports (children)
    std::set<ExpressionPort> get_connected_ports() const;

    void set_shape(const std::vector<size_t>& tensor);
    void set_layout(const std::vector<size_t>& layout);
    void set_subtensor(const std::vector<size_t>& subtensor);

    friend bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs);

private:
    std::shared_ptr<Expression> m_expr;
    Type m_type = Type::Output;
    size_t m_port_index = 0;
};
} // namespace lowered
} // namespace snippets
} // namespace ngraph
