// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/loop.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace snippets {
namespace op {

LoopBase::LoopBase(const std::vector<Output<Node>> &args) : Op(args) {}

LoopBegin::LoopBegin() : LoopBase() {
    validate_and_infer_types_except_LoopEnd();
}

void LoopBegin::validate_and_infer_types_except_LoopEnd() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 0, "LoopBegin doen't expect any inputs");
    set_output_type(0, element::f32, ov::PartialShape{ov::Shape{}});
}

void LoopBegin::validate_and_infer_types() {
    validate_and_infer_types_except_LoopEnd();
    OPENVINO_ASSERT(get_output_size() == 1, "LoopBegin must have only one output");
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "LoopBegin must have exactly one input attached to the last output");
    OPENVINO_ASSERT(ov::is_type<LoopEnd>(last_output_inputs.begin()->get_node()),
                    "LoopBegin must have LoopEnd connected to its last output");
}

std::shared_ptr<LoopEnd> LoopBegin::get_loop_end() const {
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "LoopBegin has more than one inputs attached to the last output");
    const auto& loop_end = ov::as_type_ptr<LoopEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    OPENVINO_ASSERT(loop_end != nullptr, "LoopBegin must have LoopEnd connected to its last output");
    return loop_end;
}

std::shared_ptr<Node> LoopBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<LoopBegin>();
}

LoopEnd::LoopEnd(const Output<Node>& loop_begin, size_t work_amount, size_t work_amount_increment,
                 std::vector<bool> is_incremented, std::vector<int64_t> ptr_increments, std::vector<int64_t> finalization_offsets,
                 std::vector<int64_t> element_type_sizes, size_t input_num, size_t output_num, size_t id, size_t loop_desc_id)
        : LoopBase({loop_begin}),
        m_is_incremented(std::move(is_incremented)),
        m_ptr_increments(std::move(ptr_increments)),
        m_finalization_offsets(std::move(finalization_offsets)),
        m_element_type_sizes(std::move(element_type_sizes)),
        m_work_amount(work_amount),
        m_work_amount_increment(work_amount_increment),
        m_input_num(input_num),
        m_output_num(output_num),
        m_id(id),
        m_desc_id(loop_desc_id),
        m_evaluate_once(false) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<LoopBegin> LoopEnd::get_loop_begin() {
    const auto& loop_begin = ov::as_type_ptr<LoopBegin>(get_input_source_output(get_input_size() - 1).get_node_shared_ptr());
    if (!loop_begin)
        throw std::invalid_argument("LoopEnd last input is not connected to LoopBegin");
    return loop_begin;
}

const std::vector<bool>& LoopEnd::get_is_incremented() const {
    return m_is_incremented;
}

const std::vector<int64_t>& LoopEnd::get_finalization_offsets() const {
    return m_finalization_offsets;
}

const std::vector<int64_t>& LoopEnd::get_ptr_increments() const {
    return m_ptr_increments;
}

const std::vector<int64_t>& LoopEnd::get_element_type_sizes() const {
    return m_element_type_sizes;
}

size_t LoopEnd::get_input_num() const {
    return m_input_num;
}

size_t LoopEnd::get_output_num() const {
    return m_output_num;
}

size_t LoopEnd::get_work_amount() const {
    return m_work_amount;
}

size_t LoopEnd::get_increment() const {
    return m_work_amount_increment;
}

size_t LoopEnd::get_id() const {
    return m_id;
}

size_t LoopEnd::get_desc_id() const {
    return m_desc_id;
}

bool LoopEnd::get_evaluate_once() const {
    return m_evaluate_once;
}

bool LoopEnd::has_dynamic_increments() const {
    return std::any_of(m_ptr_increments.cbegin(), m_ptr_increments.cend(), utils::is_dynamic_value<int64_t>) ||
           std::any_of(m_finalization_offsets.cbegin(), m_finalization_offsets.cend(), utils::is_dynamic_value<int64_t>);
}

void LoopEnd::set_is_incremented(std::vector<bool> is_incremented) {
    OPENVINO_ASSERT(is_incremented.size() == m_input_num + m_output_num,
                    "LoopEnd set_is_incremented is called with inconsistent is_incremented.size()");
    m_is_incremented = std::move(is_incremented);
}

void LoopEnd::set_finalization_offsets(std::vector<int64_t> offsets) {
    OPENVINO_ASSERT(offsets.size() == m_input_num + m_output_num,
                    "LoopEnd set_finalization_offsets is called with inconsistent offsets.size()");
    m_finalization_offsets = std::move(offsets);
}

void LoopEnd::set_ptr_increments(std::vector<int64_t> new_ptr_increments) {
    OPENVINO_ASSERT(new_ptr_increments.size() == m_input_num + m_output_num,
                    "LoopEnd set_ptr_increments is called with inconsistent new_ptr_increments.size()");
    m_ptr_increments = std::move(new_ptr_increments);
}

void LoopEnd::set_work_amount(size_t new_work_amount) {
    m_work_amount = new_work_amount;
}

void LoopEnd::set_evaluate_once(bool once) {
    m_evaluate_once = once;
}

void LoopEnd::set_increment(size_t new_increment) {
    m_work_amount_increment = new_increment;
}

void LoopEnd::set_id(size_t id) {
    m_id = id;
}

void LoopEnd::set_desc_id(size_t id) {
    m_desc_id = id;
}

void LoopEnd::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "LoopEnd must have one input");
    const auto loop_begin = ov::as_type_ptr<LoopBegin>(get_input_node_shared_ptr(0));
    const auto io_size = m_input_num + m_output_num;
    NODE_VALIDATION_CHECK(this, loop_begin != nullptr, "LoopEnd must have LoopBegin as the last argument");
    NODE_VALIDATION_CHECK(this, m_is_incremented.empty() || m_is_incremented.size() == io_size,
                          "is_incremented must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          io_size, " got ", m_is_incremented.size());
    NODE_VALIDATION_CHECK(this, m_ptr_increments.empty() || m_ptr_increments.size() == io_size,
                          "ptr_increments must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          io_size, " got ", m_ptr_increments.size());
    NODE_VALIDATION_CHECK(this, m_finalization_offsets.empty() || m_finalization_offsets.size() == io_size,
                          "finalization_offsets must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          io_size, " got ", m_finalization_offsets.size());
    NODE_VALIDATION_CHECK(this, m_element_type_sizes.empty() || m_element_type_sizes.size() == io_size,
                          "element_type_sizes must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          io_size, " got ", m_element_type_sizes.size());
    if (m_ptr_increments.empty())
        m_ptr_increments.resize(io_size, 0);
    if (m_finalization_offsets.empty())
        m_finalization_offsets.resize(io_size, 0);
    if (m_element_type_sizes.empty())
        m_element_type_sizes.resize(io_size, 0);
    set_output_type(0, element::f32, ov::PartialShape{});
}

bool LoopEnd::visit_attributes(AttributeVisitor &visitor) {
    std::vector<int> int_incremented(m_is_incremented.cbegin(), m_is_incremented.cend());
    visitor.on_attribute("is_incremented", int_incremented);
    visitor.on_attribute("ptr_incr", m_ptr_increments);
    visitor.on_attribute("fin_offset", m_finalization_offsets);
    visitor.on_attribute("data_sizes", m_element_type_sizes);
    visitor.on_attribute("work_amount", m_work_amount);
    visitor.on_attribute("increment", m_work_amount_increment);
    visitor.on_attribute("input_num", m_input_num);
    visitor.on_attribute("output_num", m_output_num);
    visitor.on_attribute("id", m_id);
    visitor.on_attribute("desc_id", m_desc_id);
    visitor.on_attribute("evaluate_once", m_evaluate_once);
    return true;
}

void LoopEnd::update(const lowered::RuntimeConfig::LoopDescriptor& descriptor) {
    set_work_amount(descriptor.work_amount);
    set_increment(descriptor.increment);
    set_ptr_increments(descriptor.ptr_increments);
    set_finalization_offsets(descriptor.finalization_offsets);
}

std::shared_ptr<Node> LoopEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    check_new_args_count(this, inputs);
    const auto loop_end = std::make_shared<LoopEnd>(inputs.at(0), m_work_amount, m_work_amount_increment, m_is_incremented, m_ptr_increments,
                                                    m_finalization_offsets, m_element_type_sizes, m_input_num, m_output_num, m_id, m_desc_id);
    loop_end->m_evaluate_once = m_evaluate_once;
    return loop_end;
}

} // namespace op
} // namespace snippets
} // namespace ov
