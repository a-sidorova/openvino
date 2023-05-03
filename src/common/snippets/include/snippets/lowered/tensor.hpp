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

class TensorDescriptor {
public:
    enum Type {
        Input,
        Output
    };

    TensorDescriptor() = default;
    explicit TensorDescriptor(const std::weak_ptr<Expression>& expr, Type type, size_t port,
                              const std::vector<size_t>& tensor = {}, const std::vector<size_t>& layout = {}, const std::vector<size_t>& subtensor = {});
    explicit TensorDescriptor(const std::weak_ptr<Expression>& expr, Type type, size_t port, const PortDescriptorPtr& port_desc = nullptr);

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

    friend bool operator==(const TensorDescriptor& lhs, const TensorDescriptor& rhs);
    friend bool operator!=(const TensorDescriptor& lhs, const TensorDescriptor& rhs);
    friend bool operator<(const TensorDescriptor& lhs, const TensorDescriptor& rhs);
    friend std::ostream& operator<<(std::ostream&, const TensorDescriptor& td);

private:
    std::weak_ptr<Expression> m_expr;
    Type m_type = Type::Output;
    size_t m_port_index = 0;
    PortDescriptorPtr m_port_desc;
};

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const TensorDescriptor& source_descriptor, const std::vector<TensorDescriptor>& consumer_descriptors = {});

    const TensorDescriptor& get_source() const { return m_source_port; }
    std::vector<TensorDescriptor> get_consumers() const { return m_consumer_ports; }

    void add_consumer(const TensorDescriptor& consumer);
    void remove_consumer(const TensorDescriptor& consumer);
    bool found_consumer(const TensorDescriptor& consumer) const;
    std::vector<TensorDescriptor>::const_iterator find_consumer(const TensorDescriptor& consumer) const;
    std::vector<TensorDescriptor>::iterator find_consumer(const TensorDescriptor& consumer);

    std::vector<TensorDescriptor> get_conflicted_consumers() const;
    bool is_conflicted_consumer(const TensorDescriptor& consumer) const;

    // The scheduling params of Tensor is controlled by source expression port
    std::vector<size_t> get_tensor() const { return m_source_port.get_tensor(); }
    std::vector<size_t> get_layout() const { return m_source_port.get_layout(); }
    std::vector<size_t> get_subtensor() const { return m_source_port.get_subtensor(); }

    void set_tensor(const std::vector<size_t>& tensor) { m_source_port.set_tensor(tensor); }
    void set_layout(const std::vector<size_t>& layout) { m_source_port.set_layout(layout); }
    void set_subtensor(const std::vector<size_t>& subtensor) { m_source_port.set_subtensor(subtensor); }
    void set_port_descriptor(const PortDescriptorPtr& desc) { m_source_port.set_port_descriptor(desc); }

private:
    TensorDescriptor m_source_port;
    std::vector<TensorDescriptor> m_consumer_ports;
};
using TensorPtr = std::shared_ptr<Tensor>;


} // namespace lowered
} // namespace snippets
} // namespace ngraph
