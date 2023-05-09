// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_factory.hpp"

#include "snippets/snippets_isa.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {

void LinearIR::ExpressionFactory::create_expression_inputs(const LinearIR& linear_ir, const ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr != nullptr, "Failed expression inputs creation: expression is null");
    const auto& node = expr->get_node();

    expr->m_input_tensors.resize(node->get_input_size(), nullptr);
    for (const auto& input : node->inputs()) {
        const auto input_source = input.get_source_output();
        const auto in_index = input.get_index();
        const auto& parent_expr = linear_ir.get_expr_by_node(input_source.get_node_shared_ptr());
        const auto& tensor = parent_expr->get_output_tensor(input_source.get_index());
        tensor->add_consumer(expr->get_input_port(in_index));
        expr->m_input_tensors[in_index] = tensor;
    }
}

void LinearIR::ExpressionFactory::create_expression_outputs(const LinearIR& linear_ir, const ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr != nullptr, "Failed expression outputs creation: expression is null");
    const auto& node = expr->get_node();

    expr->m_output_tensors.resize(node->get_output_size(), nullptr);
    for (const auto& output : node->outputs()) {
        const auto out_index = output.get_index();
        const auto source = expr->get_output_port(out_index);
        expr->m_output_tensors[out_index] = std::make_shared<Tensor>(source);
    }
}

// The method verifies of input tensors to availability of the expression as consumer and add it if missed
void LinearIR::ExpressionFactory::init_expression_inputs(const ExpressionPtr& expr, const std::vector<TensorPtr>& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        const auto consumers = input->get_consumers();
        const auto found = std::find_if(consumers.begin(), consumers.end(),
                                        [&](const ExpressionPort& desc) {
                                            return desc.get_index() == i && desc.get_expr() == expr;
                                        });
        if (found == consumers.end()) {
            input->add_consumer(expr->get_input_port(i));
        }
    }
    expr->m_input_tensors = inputs;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ngraph::op::v0::Parameter>& par,
                                                  const LinearIR& linear_ir, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    OPENVINO_ASSERT(model != nullptr, "To create IOExpression from Parameter there must be inited model!");
    const auto expr = std::make_shared<IOExpression>(IOExpression(par, model->get_parameter_index(par)));
    create_expression_outputs(linear_ir, expr);
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ngraph::op::v0::Result>& res,
                                                  const LinearIR& linear_ir, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    OPENVINO_ASSERT(model != nullptr, "To create IOExpression from Result there must be inited model!");
    const auto expr = std::make_shared<IOExpression>(IOExpression(res, model->get_result_index(res)));
    create_expression_inputs(linear_ir, expr);
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ov::Node>& n, const LinearIR& linear_ir,
                                                  const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(!ov::is_type<op::LoopBase>(n), "Default expression builder doesn't support LoopBegin and LoopEnd");
    // Note: ctor of shared_ptr isn't friend class for Expression
    const auto expr = std::make_shared<Expression>(Expression(n));
    create_expression_inputs(linear_ir, expr);
    create_expression_outputs(linear_ir, expr);
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<op::LoopBegin>& n, const LinearIR& linear_ir,
                                                  const std::vector<TensorPtr>& inputs) {
    OPENVINO_ASSERT(inputs.empty(), "LoopBegin cannot have inputs");
    const auto expr = std::make_shared<Expression>(Expression(n));
    init_expression_inputs(expr, inputs);
    create_expression_outputs(linear_ir, expr);
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<op::LoopEnd>& n, const LinearIR& linear_ir,
                                                  const std::vector<TensorPtr>& inputs) {
    const auto expr = std::make_shared<Expression>(Expression(n));
    // Copy port descriptor shared pointers to LoopEnd
    expr->m_input_port_descriptors.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        expr->m_input_port_descriptors[i] = inputs[i]->get_source().get_port_descriptor();
    }
    init_expression_inputs(expr, inputs);
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ov::Node>& n, const LinearIR& linear_ir,
                                                  const std::vector<TensorPtr>& inputs) {
    OPENVINO_ASSERT(!ov::is_type<ngraph::op::v0::Parameter>(n) &&
                    !ov::is_type<ngraph::op::v0::Result>(n),
                    "Expression builder with inputs doesn't support Result and Parameter");
    const auto expr = std::make_shared<Expression>(Expression(n));
    init_expression_inputs(expr, inputs);
    create_expression_outputs(linear_ir, expr);
    return expr;
}
}// namespace lowered
}// namespace snippets
}// namespace ngraph
