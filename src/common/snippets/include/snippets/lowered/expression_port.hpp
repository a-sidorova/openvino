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

class Expression;
class ExpressionPort {
public:
    enum Type {
        Input,
        Output
    };

    ExpressionPort() = default;
    explicit ExpressionPort(const std::weak_ptr<Expression>& expr, Type type, size_t port,
                            const std::vector<size_t>& tensor = {}, const std::vector<size_t>& layout = {}, const std::vector<size_t>& subtensor = {});
    explicit ExpressionPort(const std::weak_ptr<Expression>& expr, Type type, size_t port, const PortDescriptorPtr& port_desc = nullptr);

    std::shared_ptr<Expression> get_expr_ptr() const;
    const std::weak_ptr<Expression>& get_expr_wptr() const { return m_expr; }
    Type get_type() const { return m_type; }
    size_t get_index() const { return m_port_index; }

    std::vector<size_t> get_tensor() const { return m_port_desc->get_tensor(); }
    std::vector<size_t> get_layout() const { return m_port_desc->get_layout(); }
    std::vector<size_t> get_subtensor() const { return m_port_desc->get_subtensor(); }
    const PortDescriptorPtr& get_port_descriptor() const { return m_port_desc; }

    void set_tensor(const std::vector<size_t>& tensor) { m_port_desc->set_tensor(tensor); }
    void set_layout(const std::vector<size_t>& layout) { m_port_desc->set_layout(layout); }
    void set_subtensor(const std::vector<size_t>& subtensor) { m_port_desc->set_subtensor(subtensor); }
    void set_port_descriptor(const PortDescriptorPtr& desc) { m_port_desc = desc; }

    friend bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend std::ostream& operator<<(std::ostream&, const ExpressionPort& td);

private:
    std::weak_ptr<Expression> m_expr;
    Type m_type = Type::Output;
    size_t m_port_index = 0;
    PortDescriptorPtr m_port_desc;
};
} // namespace lowered
} // namespace snippets
} // namespace ngraph
