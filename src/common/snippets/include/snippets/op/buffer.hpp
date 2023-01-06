// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief The operation is for intermediate data storage
 * TODO
 *        - m_allocation_rank - rank of shape for memory allocation: shape[shape_rank - normalize(m_allocation_rank) : shape_rank].
 *                 It's needed to allocate needed memory size that depends on Tile rank, for example.
 *                 Default value is -1 (full shape)
 *        - m_static_shape - static shape that describes Buffer size in cases when Buffer doesn't have parent node.
 *        - m_element_type - element type  in cases when Buffer doesn't have parent node.
 *        - m_single - True if Buffer doesn't have parent node else False
 *        Notes:
 *               - All buffers in a graph have the same memory pointer. So if we have a few buffers,
 *                 each the corresponding MemoryAccess op for Buffer should have offset for common memory pointer of this Buffer
 *               - Buffer should be a single consumer for operation output port
 * @ingroup snippets
 */
class Buffer : public ngraph::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    BWDCMP_RTTI_DECLARATION;

    size_t get_byte_size() const;
    virtual ov::PartialShape get_allocation_shape() const = 0;

protected:
    Buffer() = default;
};

/**
 * @interface AllocationBuffer
 * @brief The operation is for allocation new empty memory
 * TODO
 * @ingroup snippets
 */
class AllocationBuffer : public Buffer {
public:
    OPENVINO_OP("AllocationBuffer", "SnippetsOpset", Buffer);
    BWDCMP_RTTI_DECLARATION;

    AllocationBuffer(const ov::Output<ov::Node>& shape, const ov::element::Type element_type);

    ov::PartialShape get_allocation_shape() const override;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

protected:
    ov::element::Type m_element_type;
};

/**
 * @interface IntermediateBuffer
 * @brief The operation is for intermediate data storage
 * TODO
 * @ingroup snippets
 */
class IntermediateBuffer : public Buffer {
public:
    OPENVINO_OP("IntermediateBuffer", "SnippetsOpset", Buffer);
    BWDCMP_RTTI_DECLARATION;

    IntermediateBuffer(const ov::Output<ov::Node>& x);
    IntermediateBuffer(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& shape);

    ov::PartialShape get_allocation_shape() const override;

    bool visit_attributes(AttributeVisitor& visitor) override { return true; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    static std::shared_ptr<ov::Node> create_shape_constant(const ov::PartialShape& shape, size_t allocation_rank);
    static std::shared_ptr<ov::Node> create_shape_constant(const ov::PartialShape& shape);
};

} // namespace op
} // namespace snippets
} // namespace ngraph
