// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets/op/memory_access.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace op {
namespace {
// Note:
//   - If `is_forward` is True, `result shape` is ordered `shape` by `layout`
//   - If `is_forward` is False, `result shape` is original shape to which the `layout` was applied
ov::PartialShape get_pshape(const ov::PartialShape& shape, const std::vector<size_t>& layout, bool is_forward) {
    if (layout.empty())
        return shape;
    ov::PartialShape reordered_shape(std::vector<Dimension>(layout.size()));
    if (shape.rank().is_dynamic())
        OPENVINO_THROW("get_reordered_planar_shape can't be called for outputs with dynamic rank");
    const size_t rank = shape.rank().get_length();
    if (layout.size() > rank)
        OPENVINO_THROW("Layout rank can't be larger than tensor rank");
    // Note that it can be smaller though, for example tensor shape can be prepended with 1 for scheduling purposes
    if (std::any_of(layout.begin(), layout.end(), [=](size_t x) {return x >= rank;}))
        OPENVINO_THROW("Invalid layout detected: all layout indexes must be smaller than the tensor rank");
    utils::ordered_vector(shape, layout, is_forward, reordered_shape);
    return reordered_shape;
}
}  // namespace

MemoryAccess::MemoryAccess(const OutputVector& arguments, size_t input_count, size_t output_count) : Op(arguments) {
    auto init_iota_set = [](size_t num) {
        if (num == 0)
            return std::set<size_t>{};
        std::vector<size_t> vec(num);
        std::iota(vec.begin(), vec.end(), 0);
        return std::set<size_t>(vec.begin(), vec.end());
    };
    ctor_initialize(init_iota_set(input_count), init_iota_set(output_count));
}

MemoryAccess::MemoryAccess(const OutputVector& arguments, const std::set<size_t>& input_ports, const std::set<size_t>& output_ports) : Op(arguments) {
    ctor_initialize(input_ports, output_ports);
}

MemoryAccess::MemoryAccess(const OutputVector& arguments, const PortMap& input_ports, const PortMap& output_ports)
    : Op(arguments), m_input_ports(input_ports), m_output_ports(output_ports) {}

void MemoryAccess::ctor_initialize(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports) {
    for (auto port : input_ports) {
        m_input_ports[port] = PortDescriptor();
    }
    for (auto port : output_ports) {
        m_output_ports[port] = PortDescriptor();
    }
}

bool MemoryAccess::is_full_memory_access_op() const {
    for (size_t i = 0; i < get_input_size(); ++i) {
        if (!is_memory_access_input_port(i))
            return false;
    }
    for (size_t i = 0; i < get_output_size(); ++i) {
        if (!is_memory_access_output_port(i))
            return false;
    }
    return true;
}

bool MemoryAccess::visit_attributes(AttributeVisitor& visitor) {
    for (const auto& p : m_input_ports) {
        auto idx = p.first;
        auto port = p.second;
        visitor.on_attribute("count_in_" + std::to_string(idx), port.count);
        visitor.on_attribute("offset_in_" + std::to_string(idx), port.offset);
        visitor.on_attribute("order_in_" + std::to_string(idx), port.order);
    }
    for (const auto& p : m_output_ports) {
        auto idx = p.first;
        auto port = p.second;
        visitor.on_attribute("count_out_" + std::to_string(idx), port.count);
        visitor.on_attribute("offset_out_" + std::to_string(idx), port.offset);
        visitor.on_attribute("order_out_" + std::to_string(idx), port.order);
    }
    return true;
}

bool MemoryAccess::is_memory_access_input_port(size_t idx) const {
    return m_input_ports.find(idx) != m_input_ports.end();
}
bool MemoryAccess::is_memory_access_output_port(size_t idx) const {
    return m_output_ports.find(idx) != m_output_ports.end();
}

void MemoryAccess::set_input_port_descriptor(const PortDescriptor& desc, const size_t i) {
    const auto it = m_input_ports.find(i);
    OPENVINO_ASSERT(it != m_input_ports.end(), "Index of input port descriptor should be less than count of input ports");
    (*it).second = desc;
}

void MemoryAccess::set_output_port_descriptor(const PortDescriptor& desc, const size_t i) {
    const auto it = m_output_ports.find(i);
    OPENVINO_ASSERT(it != m_output_ports.end(), "Index of output port descriptor should be less than count of output ports");
    (*it).second = desc;
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_input_port_descriptor(const size_t i) const {
    const auto it = m_input_ports.find(i);
    OPENVINO_ASSERT(it != m_input_ports.end(), "Index of input port descriptor should be less than count of input ports");
    return (*it).second;
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_output_port_descriptor(const size_t i) const {
    const auto it = m_output_ports.find(i);
    OPENVINO_ASSERT(it != m_output_ports.end(), "Index of output port descriptor should be less than count of output ports");
    return (*it).second;
}

void MemoryAccess::set_input_count(size_t count, size_t idx) {
    OPENVINO_ASSERT(m_input_ports.count(idx), "Index of input port descriptor should be less than count of input ports");
    m_input_ports[idx].count = count;
}
void MemoryAccess::set_output_count(size_t count, size_t idx) {
    OPENVINO_ASSERT(m_input_ports.count(idx), "Index of output port descriptor should be less than count of output ports");
    m_output_ports[idx].count = count;
}
void MemoryAccess::set_input_offset(size_t offset, size_t idx) {
    OPENVINO_ASSERT(m_input_ports.count(idx), "Index of input port descriptor should be less than count of input ports");
    m_input_ports[idx].offset = offset;
}
void MemoryAccess::set_output_offset(size_t offset, size_t idx) {
    OPENVINO_ASSERT(m_input_ports.count(idx), "Index of output port descriptor should be less than count of output ports");
    m_output_ports[idx].offset = offset;
}
void MemoryAccess::set_input_order(std::vector<size_t> order, size_t idx) {
    OPENVINO_ASSERT(m_input_ports.count(idx), "Index of input port descriptor should be less than count of input ports");
    m_input_ports[idx].order = std::move(order);
}
void MemoryAccess::set_output_order(std::vector<size_t> order, size_t idx) {
    OPENVINO_ASSERT(m_input_ports.count(idx), "Index of output port descriptor should be less than count of output ports");
    m_output_ports[idx].order = std::move(order);
}
size_t MemoryAccess::get_input_count(size_t idx) const {
    return get_input_port_descriptor(idx).count;
}
size_t MemoryAccess::get_output_count(size_t idx) const {
    return get_output_port_descriptor(idx).count;
}
size_t MemoryAccess::get_input_offset(size_t idx) const {
    return get_input_port_descriptor(idx).offset;
}
size_t MemoryAccess::get_output_offset(size_t idx) const {
    return get_output_port_descriptor(idx).offset;
}
const std::vector<size_t>& MemoryAccess::get_input_order(size_t idx) const {
    return get_input_port_descriptor(idx).order;
}
const std::vector<size_t>& MemoryAccess::get_output_order(size_t idx) const {
     return get_output_port_descriptor(idx).order;
}

ov::PartialShape MemoryAccess::get_input_planar_partial_shape(size_t idx) const {
    return get_pshape(get_input_partial_shape(idx), get_input_order(idx), true);
}
ov::PartialShape MemoryAccess::get_output_preordered_partial_shape(size_t idx) const {
    return get_pshape(get_output_partial_shape(idx), get_output_order(idx), false);
}

} // namespace op
} // namespace snippets
} // namespace ov
