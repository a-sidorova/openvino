// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief This is a base class for memory storage.
 *        Notes:
 *               - All buffers with the same reg_group in a graph have the same memory pointer. So if we have a few buffers,
 *                 each the corresponding MemoryAccess op for Buffer should have offset for common memory pointer of this Buffer
 *               - Buffer should be a single consumer for operation output port
 * @param m_shape - output allocation shape for Buffer with type NewMemory
 * @param m_reg_group - number of register group. The Buffers from the same group will have the same GPR
 * @ingroup snippets
 */
class Buffer : public ov::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    Buffer() = default;
    Buffer(const OutputVector& arguments, const ov::Shape& shape, size_t reg_group = 0, ov::element::Type element_type = ov::element::u8);

    bool visit_attributes(AttributeVisitor& visitor) override;

    size_t get_reg_group() const { return m_reg_group; }
    const ov::Shape& get_allocation_shape() const { return m_shape; }
    size_t get_byte_size() const;

    void set_reg_group(size_t reg_group) { m_reg_group = reg_group; }
    void set_allocation_shape(const ov::Shape& allocation_shape) { m_shape = allocation_shape; }

protected:
    ov::Shape m_shape = {};
    size_t m_reg_group = 0;
    ov::element::Type m_element_type = ov::element::u8;  // u8 - default 1 byte
};

/**
 * @interface IntermediateMemoryBuffer
 * @brief Represents an intermediate memory storage operation. It always has a parent.
 * @ingroup snippets
 *
 */
class IntermediateMemoryBuffer : public Buffer {
public:
    OPENVINO_OP("IntermediateMemoryBuffer", "SnippetsOpset", Buffer);
    IntermediateMemoryBuffer() = default;
    IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, const ov::Shape& shape, size_t reg_group = 0);
    IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, int32_t allocation_rank = -1, size_t reg_group = 0);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    ov::Shape compute_shape_from_allocation_rank(const ov::Output<ov::Node>& arg, int32_t allocation_rank);
};

/**
 * @interface NewMemoryBuffer
 * @brief Represents a new empty memory for allocation with specified shape. It has no parent operations.
 * @ingroup snippets
 *
 */
class NewMemoryBuffer : public Buffer {
public:
    OPENVINO_OP("NewMemoryBuffer", "SnippetsOpset", Buffer);
    NewMemoryBuffer() = default;
    NewMemoryBuffer(const ov::Shape& shape, size_t reg_group = 0, ov::element::Type element_type = ov::element::u8);

    void validate_and_infer_types() override;
    void set_element_type(ov::element::Type element_type);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    class ShapeInfer : public IShapeInferSnippets {
        ov::Shape m_shape;
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };
};

} // namespace op
} // namespace snippets
} // namespace ov
