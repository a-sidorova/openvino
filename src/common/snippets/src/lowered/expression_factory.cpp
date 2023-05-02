// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_factory.hpp"

#include "snippets/snippets_isa.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {

ExpressionPtr LinearIR::BaseExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    OPENVINO_THROW("The Factory doesn't support default builder");
}
ExpressionPtr LinearIR::BaseExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                                     const std::vector<TensorPtr> inputs) {
    OPENVINO_THROW("The Factory doesn't support builder with just input tensors");
}
ExpressionPtr LinearIR::BaseExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                                     const std::vector<TensorPtr> inputs, const std::vector<TensorPtr> outputs) {
    OPENVINO_THROW("The Factory doesn't support builder with input and outputs tensors");
}

std::shared_ptr<LinearIR::BaseExpressionFactory> LinearIR::BaseExpressionFactory::get(const LinearIR& linear_ir, const std::shared_ptr<Node>& n) {
    if (ov::is_type<ov::op::v0::Parameter>(n)) {
        return std::make_shared<LinearIR::ParameterExpressionFactory>(linear_ir);
    }
    if (ov::is_type<ov::op::v0::Result>(n)) {
        return std::make_shared<LinearIR::ResultExpressionFactory>(linear_ir);
    }
    if (ov::is_type<op::LoopBegin>(n)) {
        return std::make_shared<LinearIR::LoopBeginExpressionFactory>(linear_ir);
    }
    if (ov::is_type<op::LoopEnd>(n)) {
        return std::make_shared<LinearIR::LoopEndExpressionFactory>(linear_ir);
    }
    return std::make_shared<LinearIR::ExpressionFactory>(linear_ir);
}

std::vector<TensorPtr> LinearIR::BaseExpressionFactory::create_expression_inputs(const ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr != nullptr, "Failed expression inputs creation: expression is null");
    const auto& node = expr->get_node();

    std::vector<TensorPtr> inputs(node->get_input_size(), nullptr);
    for (const auto& input : node->inputs()) {
        const auto input_source = input.get_source_output();
        const auto in_index = input.get_index();
        const auto out_index = input_source.get_index();
        const auto parent = input_source.get_node_shared_ptr();
        const auto parent_expr = m_linear_ir.get_expr_by_node(parent);
        const auto tensor = parent_expr->get_outputs()[out_index];
        const auto tensor_desc = TensorDescriptor(expr, TensorDescriptor::Type::Input, in_index, PortManager::get_port_descriptor_ptr(input));
        tensor->add_consumer(tensor_desc);
        inputs[in_index] = tensor;
    }
    return inputs;
}

std::vector<TensorPtr> LinearIR::BaseExpressionFactory::create_expression_outputs(const ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr != nullptr, "Failed expression outputs creation: expression is null");
    const auto& node = expr->get_node();

    std::vector<TensorPtr> outputs(node->get_output_size(), nullptr);
    for (const auto& output : node->outputs()) {
        const auto out_index = output.get_index();
        const auto tensor_desc = TensorDescriptor(expr, TensorDescriptor::Type::Output, out_index, PortManager::get_port_descriptor_ptr(output));
        outputs[out_index] = std::make_shared<Tensor>(tensor_desc);
    }
    return outputs;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression
    return std::make_shared<Expression>(Expression(n));
}

ExpressionPtr LinearIR::ExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    const auto expr = create(n, model);
    expr->init_inputs(create_expression_inputs(expr));
    expr->init_outputs(create_expression_outputs(expr));
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                                 const std::vector<TensorPtr> inputs) {
    const auto expr = create(n, model);
    expr->init_inputs_with_validation(inputs);
    expr->init_outputs(create_expression_outputs(expr));
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                                 const std::vector<TensorPtr> inputs, const std::vector<TensorPtr> outputs) {
    const auto expr = create(n, model);
    expr->init_inputs_with_validation(inputs);
    expr->init_outputs(outputs);
    return expr;
}

ExpressionPtr LinearIR::ParameterExpressionFactory::create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    if (const auto& par = as_type_ptr<opset1::Parameter>(n)) {
        OPENVINO_ASSERT(model != nullptr, "To create IOExpression from Parameter there must be inited model!");
        return std::make_shared<IOExpression>(IOExpression(par, model->get_parameter_index(par)));
    }
    OPENVINO_THROW("ParameterExpressionFactory support only Parameter node");
}

ExpressionPtr LinearIR::ParameterExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    const auto expr = create(n, model);
    expr->init_inputs({});
    expr->init_outputs(create_expression_outputs(expr));
    return expr;
}

ExpressionPtr LinearIR::ResultExpressionFactory::create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    if (const auto& res = as_type_ptr<opset1::Result>(n)) {
        OPENVINO_ASSERT(model != nullptr, "To create IOExpression from Result there must be inited model!");
        return std::make_shared<IOExpression>(IOExpression(res, model->get_result_index(res)));
    }
    OPENVINO_THROW("ResultExpressionFactory support only Result node");
}

ExpressionPtr LinearIR::ResultExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    const auto expr = create(n, model);
    expr->init_inputs(create_expression_inputs(expr));
    expr->init_outputs({});
    return expr;
}

ExpressionPtr LinearIR::LoopBeginExpressionFactory::create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    if (const auto& op = as_type_ptr<op::LoopBegin>(n)) {
        return std::make_shared<Expression>(Expression(op));
    }
    OPENVINO_THROW("LoopBeginExpressionFactory support only LoopBegin node");
}

ExpressionPtr LinearIR::LoopBeginExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                                          const std::vector<TensorPtr> inputs) {
    OPENVINO_ASSERT(inputs.empty(), "LoopBegin cannot have inputs");
    const auto expr = create(n, model);
    expr->init_inputs(inputs);
    expr->init_outputs(create_expression_outputs(expr));
    return expr;
}

ExpressionPtr LinearIR::LoopEndExpressionFactory::create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    if (const auto& op = as_type_ptr<op::LoopEnd>(n)) {
        return std::make_shared<Expression>(Expression(op));
    }
    OPENVINO_THROW("LoopEndExpressionFactory support only LoopEnd node");
}

ExpressionPtr LinearIR::LoopEndExpressionFactory::build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                                        const std::vector<TensorPtr> inputs) {
    const auto expr = create(n, model);
    expr->init_inputs_with_validation(inputs);
    expr->init_outputs({});
    return expr;
}


}// namespace lowered
}// namespace snippets
}// namespace ngraph
