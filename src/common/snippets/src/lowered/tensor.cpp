// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/tensor.hpp"

#include <snippets/itt.hpp>
#include "snippets/utils.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

Tensor::Tensor(const ExpressionPort& source_descriptor, const std::vector<ExpressionPort>& consumer_descriptors)
    : m_source_port(source_descriptor), m_consumer_ports(consumer_descriptors) {}

std::vector<ExpressionPort>::const_iterator Tensor::find_consumer(const ExpressionPort& consumer) const {
    // Note: Find by shared ptr and index port is enough since these parameters must be unique
    return std::find_if(m_consumer_ports.begin(), m_consumer_ports.end(),
                        [&consumer](const ExpressionPort& td) {
                            return consumer.get_expr_ptr() == td.get_expr_ptr() && consumer.get_index() == td.get_index();
                        });
}

std::vector<ExpressionPort>::iterator Tensor::find_consumer(const ExpressionPort& consumer) {
    // Note: Find by shared ptr and index port is enough since these parameters must be unique
    return std::find_if(m_consumer_ports.begin(), m_consumer_ports.end(),
                        [&consumer](const ExpressionPort& td) {
                            return consumer.get_expr_ptr() == td.get_expr_ptr() && consumer.get_index() == td.get_index();
                        });
}

bool Tensor::found_consumer(const ExpressionPort& consumer) const {
    return find_consumer(consumer) != m_consumer_ports.end();
}

void Tensor::add_consumer(const ExpressionPort& consumer) {
    OPENVINO_ASSERT(!found_consumer(consumer), "Consumer has been already added to Tensor!");
    m_consumer_ports.push_back(consumer);
}

void Tensor::remove_consumer(const ExpressionPort& consumer) {
    const auto& found = find_consumer(consumer);
    OPENVINO_ASSERT(found != m_consumer_ports.end(), "Consumer is missed in Tensor!");
    m_consumer_ports.erase(found);
}

std::vector<ExpressionPort> Tensor::get_conflicted_consumers() const {
    std::vector<ExpressionPort> conflicted_consumers;
    for (const auto& consumer : m_consumer_ports) {
        if (is_conflicted_consumer(consumer)) {
            conflicted_consumers.push_back(consumer);
        }
    }
    return conflicted_consumers;
}

bool Tensor::is_conflicted_consumer(const ExpressionPort& consumer) const {
    OPENVINO_ASSERT(found_consumer(consumer), "Failed check for conflicted consumer: it's not a consumer fot the Tensor");
    return get_tensor() != consumer.get_tensor() ||
           get_layout() != consumer.get_layout() ||
           get_subtensor() != consumer.get_subtensor();
}

bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    if (&rhs == &lhs)
        return true;
    return lhs.m_type == rhs.m_type &&
           lhs.m_expr.lock() == rhs.m_expr.lock() &&
           lhs.m_port_index == rhs.m_port_index &&
           lhs.m_port_desc == rhs.m_port_desc;
}
bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    return !(lhs == rhs);
}
bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "ExpressionPorts must be of the same type for comparison!");
    return lhs.get_index() < rhs.get_index() &&
           lhs.get_expr_ptr() < rhs.get_expr_ptr() &&
           lhs.get_tensor() < rhs.get_tensor() &&
           lhs.get_layout() < rhs.get_layout() &&
           lhs.get_subtensor() < rhs.get_subtensor();
}

std::ostream& operator<<(std::ostream& ss, const ExpressionPort& td) {
    auto print_vector = [&ss](const std::vector<size_t>& data){
        ss << "[";
        for (auto i : data)
            ss << i << ",";
        ss << (data.empty() ? "]" : "\b]");
    };
    ss  << "{Tensor: ";
    print_vector(td.get_tensor());
    ss  << " Subtensor: ";
    print_vector(td.get_subtensor());
    ss  << " Layout: ";
    print_vector(td.get_layout());
    ss << "}";
    return ss;
}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
