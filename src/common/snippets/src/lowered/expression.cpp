// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression.hpp"

#include <snippets/itt.hpp>
#include "snippets/utils.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {
namespace lowered {

size_t Expression::LOOP_NULL_ID = SIZE_MAX;

Expression::Expression(const std::shared_ptr<Node>& n)
    : m_source_node{n}, m_emitter{nullptr}, m_inputs{}, m_outputs{}, m_reg_info{{}, {}} {}

const TensorPtr& Expression::input(size_t i) const {
    OPENVINO_ASSERT(i < m_inputs.size(), "Failed to get input: target input port must be less than input count!");
    return m_inputs[i];
}
const TensorPtr& Expression::output(size_t i) const {
    OPENVINO_ASSERT(i < m_outputs.size(), "Failed to get output: target output port must be less than output count!");
    return m_outputs[i];
}

std::shared_ptr<Node> Expression::get_node() const {
    if (!m_source_node)
        OPENVINO_THROW("An attempt to get uninitialized node from lowered expression");
    return  m_source_node;
}

std::shared_ptr<Emitter> Expression::get_emitter() const {
    return m_emitter;
}

void Expression::init_emitter(const std::shared_ptr<const TargetMachine>& target) {
    m_emitter = target->get(m_source_node->get_type_info())(m_source_node);
}

void Expression::replace_input(size_t port, TensorPtr to) {
    OPENVINO_ASSERT(port < m_inputs.size(), "Failed to replace: target input port must be less than input count!");
    m_inputs[port] = std::move(to);
}

void Expression::replace_output(size_t port, TensorPtr to) {
    OPENVINO_ASSERT(port < m_outputs.size(), "Failed to replace: target output port must be less than output count!");
    m_outputs[port] = std::move(to);
}

void Expression::set_loop_id(size_t id, size_t idx) {
    if (id != LOOP_NULL_ID) {
        OPENVINO_ASSERT((std::find(m_loop_ids.begin(), m_loop_ids.end(), id) == m_loop_ids.end()),
                        "Expression cannot have several the same Loops");
    }
    if (m_loop_ids.size() <= idx) {
        m_loop_ids.resize(idx + 1, LOOP_NULL_ID);
    }
    m_loop_ids[idx] = id;
}

void Expression::remove_loop_id(size_t id) {
    auto it = std::find(m_loop_ids.begin(), m_loop_ids.end(), id);
    OPENVINO_ASSERT(it == m_loop_ids.end(), "Expression doesn't have the Loop with ID " + std::to_string(id));
    *it = Expression::LOOP_NULL_ID;
}

ExpressionPort Expression::input_port(size_t i) {
    OPENVINO_ASSERT(i < m_inputs.size(), "Failed to get input port: target input port must be less than input count!");
    const auto& input = m_inputs[i];
    const auto consumers = input->get_consumers();
    const auto found = std::find_if(consumers.begin(), consumers.end(),
                                    [&](const ExpressionPort& desc) {
                                                return desc.get_index() == i && desc.get_expr_ptr() == this->shared_from_this();
                                          });
    OPENVINO_ASSERT(found != consumers.end(), "Input ExpressionPort for Expression hasn't found in input Tensor!");
    return *found;
}

ExpressionPort Expression::output_port(size_t i) {
    OPENVINO_ASSERT(i < m_outputs.size(), "Failed to get output port: target output port must be less than output count!");
    return m_outputs[i]->get_source();
}

IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Parameter>& par, int64_t index)
        : Expression(par), m_index(index), m_type{io_type::INPUT} {}
IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Result>& res, int64_t index)
        : Expression(res), m_index(index), m_type{io_type::OUTPUT} {}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
