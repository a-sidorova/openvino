// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "snippets/port_descriptor.hpp"

#include "expression_port.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

class Expression;

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const ExpressionPort& source_descriptor, const std::vector<ExpressionPort>& consumer_descriptors = {});

    const ExpressionPort& get_source() const { return m_source_port; }
    std::vector<ExpressionPort> get_consumers() const { return m_consumer_ports; }

    void add_consumer(const ExpressionPort& consumer);
    void remove_consumer(const ExpressionPort& consumer);
    bool found_consumer(const ExpressionPort& consumer) const;
    std::vector<ExpressionPort>::const_iterator find_consumer(const ExpressionPort& consumer) const;
    std::vector<ExpressionPort>::iterator find_consumer(const ExpressionPort& consumer);

    std::vector<ExpressionPort> get_conflicted_consumers() const;
    bool is_conflicted_consumer(const ExpressionPort& consumer) const;

    // The scheduling params of Tensor is controlled by source expression port
    std::vector<size_t> get_tensor() const { return m_source_port.get_tensor(); }
    std::vector<size_t> get_layout() const { return m_source_port.get_layout(); }
    std::vector<size_t> get_subtensor() const { return m_source_port.get_subtensor(); }

    void set_tensor(const std::vector<size_t>& tensor) { m_source_port.set_tensor(tensor); }
    void set_layout(const std::vector<size_t>& layout) { m_source_port.set_layout(layout); }
    void set_subtensor(const std::vector<size_t>& subtensor) { m_source_port.set_subtensor(subtensor); }
    void set_port_descriptor(const PortDescriptorPtr& desc) { m_source_port.set_port_descriptor(desc); }

private:
    ExpressionPort m_source_port;
    std::vector<ExpressionPort> m_consumer_ports;
};
using TensorPtr = std::shared_ptr<Tensor>;


} // namespace lowered
} // namespace snippets
} // namespace ngraph
