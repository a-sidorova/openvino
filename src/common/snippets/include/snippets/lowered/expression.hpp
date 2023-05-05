// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>

#include "snippets/emitter.hpp"
#include "snippets/target_machine.hpp"
#include "snippets/lowered/tensor.hpp"


namespace ngraph {
namespace snippets {
namespace lowered {

class LinearIR;

class Expression : public std::enable_shared_from_this<Expression> {
    friend class LinearIR;

public:
    static size_t LOOP_NULL_ID;

    Expression() = default;
    virtual ~Expression() = default;

    std::shared_ptr<Node> get_node() const;
    std::shared_ptr<Emitter> get_emitter() const;

    RegInfo get_reg_info() const { return  m_reg_info; }
    void set_reg_info(RegInfo rinfo) { m_reg_info = std::move(rinfo); }

    const TensorPtr& input(size_t i) const;
    const TensorPtr& output(size_t i) const;
    const std::vector<TensorPtr>& inputs() const { return m_inputs; }
    const std::vector<TensorPtr>& outputs() const { return m_outputs; }
    size_t get_input_count() const { return m_inputs.size(); }
    size_t get_output_count() const { return m_outputs.size(); }

    std::vector<size_t> get_loop_ids() const { return m_loop_ids; }
    void set_loop_ids(const std::vector<size_t>& loops) { m_loop_ids = loops; }
    void set_loop_id(size_t id, size_t idx);
    void remove_loop_id(size_t id);

    void init_emitter(const std::shared_ptr<const TargetMachine>& target);

    TensorDescriptor input_port(size_t i);
    TensorDescriptor output_port(size_t i);

protected:
    // Note: The constructor and tensor initialization are private since an expression can be created only by Linear IR.
    //       These methods must be used only by Linear IR builder of expressions!
    explicit Expression(const std::shared_ptr<Node>& n);
    void init_inputs(const std::vector<TensorPtr>& inputs) { m_inputs = inputs; }
    void init_outputs(const std::vector<TensorPtr>& outputs) { m_outputs = outputs; }

    // Note: These methods don't control availability of the current expression in this Tensor (as Consumer or Source)
    void replace_input(size_t port, TensorPtr to);
    void replace_output(size_t port, TensorPtr to);

    std::shared_ptr<Node> m_source_node{nullptr};
    std::shared_ptr<Emitter> m_emitter{nullptr};
    std::vector<TensorPtr> m_inputs;
    std::vector<TensorPtr> m_outputs;
    RegInfo m_reg_info{{}, {}};
    // The order Loops identifies: Outer ---> Inner
    std::vector<size_t> m_loop_ids;
};
using ExpressionPtr = std::shared_ptr<Expression>;

class IOExpression : public Expression {
    friend class LinearIR;

public:
    enum class io_type {INPUT, OUTPUT, UNDEFINED};

    int64_t get_index() const  { return m_index; }
    io_type get_type() const { return m_type; }

private:
    explicit IOExpression(const std::shared_ptr<ov::opset1::Parameter>& n, int64_t index);
    explicit IOExpression(const std::shared_ptr<ov::opset1::Result>& n, int64_t index);

    int64_t m_index = -1;
    io_type m_type = io_type::UNDEFINED;
};

} // namespace lowered
} // namespace snippets
} // namespace ngraph
