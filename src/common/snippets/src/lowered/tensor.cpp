// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/tensor.hpp"

#include <snippets/itt.hpp>
#include "snippets/utils.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

TensorDescriptor::TensorDescriptor(const std::weak_ptr<Expression>& expr, Type type, size_t port,
                                   const std::vector<size_t>& tensor, const std::vector<size_t>& layout, const std::vector<size_t>& subtensor)
    : m_expr(expr), m_type(type), m_port_index(port), m_port_desc(std::make_shared<PortDescriptor>(tensor, subtensor, layout)) {}

TensorDescriptor::TensorDescriptor(const std::weak_ptr<Expression>& expr, Type type, size_t port, const PortDescriptorPtr& port_desc)
    : m_expr(expr), m_type(type), m_port_index(port) {
    PortDescriptorPtr local_port_desc = port_desc;
    if (!local_port_desc) {
        if (type == Type::Input) {
            local_port_desc = PortManager::get_port_descriptor_ptr(expr.lock()->get_node()->input(port));
        } else if (type == Type::Output) {
            local_port_desc = PortManager::get_port_descriptor_ptr(expr.lock()->get_node()->output(port));
        } else {
            OPENVINO_THROW("TensorDescriptor supports only Input and Output type!");
        }
    }

    m_port_desc = local_port_desc;
}

std::shared_ptr<Expression> TensorDescriptor::get_expr_ptr() const {
    auto shared = m_expr.lock();
    OPENVINO_ASSERT(shared != nullptr, "Failed attempt to get shared pointer of source expression: nullptr");
    return shared;
}

Tensor::Tensor(const TensorDescriptor& source_descriptor, const std::vector<TensorDescriptor>& consumer_descriptors)
    : m_source_port(source_descriptor), m_consumer_ports(consumer_descriptors) {}

std::vector<TensorDescriptor>::const_iterator Tensor::find_consumer(const TensorDescriptor& consumer) const {
    // Note: Find by shared ptr and index port is enough since these parameters must be unique
    return std::find_if(m_consumer_ports.begin(), m_consumer_ports.end(),
                        [&consumer](const TensorDescriptor& td) {
                            return consumer.get_expr_ptr().get() == td.get_expr_ptr().get() && consumer.get_index() == td.get_index();
                        });
}

std::vector<TensorDescriptor>::iterator Tensor::find_consumer(const TensorDescriptor& consumer) {
    // Note: Find by shared ptr and index port is enough since these parameters must be unique
    return std::find_if(m_consumer_ports.begin(), m_consumer_ports.end(),
                        [&consumer](const TensorDescriptor& td) {
                            return consumer.get_expr_ptr().get() == td.get_expr_ptr().get() && consumer.get_index() == td.get_index();
                        });
}

bool Tensor::found_consumer(const TensorDescriptor& consumer) const {
    return find_consumer(consumer) != m_consumer_ports.end();
}

void Tensor::add_consumer(const TensorDescriptor& consumer) {
    OPENVINO_ASSERT(!found_consumer(consumer), "Consumer has been already added to Tensor!");
    m_consumer_ports.push_back(consumer);
}

void Tensor::remove_consumer(const TensorDescriptor& consumer) {
    const auto& found = find_consumer(consumer);
    OPENVINO_ASSERT(found != m_consumer_ports.end(), "Consumer is missed in Tensor!");
    m_consumer_ports.erase(found);
}

std::vector<TensorDescriptor> Tensor::get_conflicted_consumers() const {
    std::vector<TensorDescriptor> conflicted_consumers;
    for (const auto& consumer : m_consumer_ports) {
        if (is_conflicted_consumer(consumer)) {
            conflicted_consumers.push_back(consumer);
        }
    }
    return conflicted_consumers;
}

bool Tensor::is_conflicted_consumer(const TensorDescriptor& consumer) const {
    return get_tensor() != consumer.get_tensor() ||
           get_layout() != consumer.get_layout() ||
           get_subtensor() != consumer.get_subtensor();
}

bool operator==(const TensorDescriptor& lhs, const TensorDescriptor& rhs) {
    if (&rhs == &lhs)
        return true;
    return lhs.m_type == rhs.m_type &&
           lhs.m_expr.lock().get() == rhs.m_expr.lock().get() &&
           lhs.m_port_index == rhs.m_port_index &&
           lhs.m_port_desc == rhs.m_port_desc;
}
bool operator!=(const TensorDescriptor& lhs, const TensorDescriptor& rhs) {
    return !(lhs == rhs);
}
bool operator<(const TensorDescriptor& lhs, const TensorDescriptor& rhs) {
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "TensorDescriptors must be of the same type for comparison!");
    return lhs.get_index() < rhs.get_index() &&
           lhs.get_expr_ptr().get() < rhs.get_expr_ptr().get() &&
           lhs.get_tensor() < rhs.get_tensor() &&
           lhs.get_layout() < rhs.get_layout() &&
           lhs.get_subtensor() < rhs.get_subtensor();
}

std::ostream& operator<<(std::ostream& ss, const TensorDescriptor& td) {
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
