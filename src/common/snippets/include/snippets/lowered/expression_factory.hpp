// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_ir.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {

class LinearIR::BaseExpressionFactory {
public:
    BaseExpressionFactory() = default;
    BaseExpressionFactory(const LinearIR& linear_ir) : m_linear_ir(linear_ir) {}

    virtual ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model);
    virtual ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                const std::vector<TensorPtr>& inputs);
    virtual ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                                const std::vector<TensorPtr>& inputs, const std::vector<TensorPtr>& outputs);

    static std::shared_ptr<LinearIR::BaseExpressionFactory> get(const LinearIR& linear_ir, const std::shared_ptr<Node>& n);

protected:
    virtual ExpressionPtr create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) = 0;
    // Creates inputs for expression using parent output tensors
    virtual std::vector<TensorPtr> create_expression_inputs(const ExpressionPtr& expr);
    // Creates new output tensors
    virtual std::vector<TensorPtr> create_expression_outputs(const ExpressionPtr& expr);
    // The method verifies of input tensors to availability of the expression as consumer and add it if missed
    virtual void validate_inputs(const ExpressionPtr& expr, const std::vector<TensorPtr>& inputs);

    LinearIR m_linear_ir;
};

class LinearIR::ExpressionFactory : public LinearIR::BaseExpressionFactory {
public:
    ExpressionFactory() = default;
    ExpressionFactory(const LinearIR& linear_ir) : BaseExpressionFactory(linear_ir) {}

    ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;
    ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                        const std::vector<TensorPtr>& inputs) override;
    ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                        const std::vector<TensorPtr>& inputs, const std::vector<TensorPtr>& outputs) override;

protected:
    ExpressionPtr create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;
};

class LinearIR::ParameterExpressionFactory : public LinearIR::BaseExpressionFactory {
public:
    ParameterExpressionFactory() = default;
    ParameterExpressionFactory(const LinearIR& linear_ir) : BaseExpressionFactory(linear_ir) {}

    ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;

protected:
    ExpressionPtr create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;
};

class LinearIR::ResultExpressionFactory : public LinearIR::BaseExpressionFactory {
public:
    ResultExpressionFactory() = default;
    ResultExpressionFactory(const LinearIR& linear_ir) : BaseExpressionFactory(linear_ir) {}

    ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;

protected:
    ExpressionPtr create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;
};

class LinearIR::LoopBeginExpressionFactory : public LinearIR::BaseExpressionFactory {
public:
    LoopBeginExpressionFactory() = default;
    LoopBeginExpressionFactory(const LinearIR& linear_ir) : BaseExpressionFactory(linear_ir) {}

    ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                        const std::vector<TensorPtr>& inputs) override;

protected:
    ExpressionPtr create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;
};

class LinearIR::LoopEndExpressionFactory : public LinearIR::BaseExpressionFactory {
public:
    LoopEndExpressionFactory() = default;
    LoopEndExpressionFactory(const LinearIR& linear_ir) : BaseExpressionFactory(linear_ir) {}

    ExpressionPtr build(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model,
                        const std::vector<TensorPtr>& inputs) override;

protected:
    ExpressionPtr create(const std::shared_ptr<Node>& n, const std::shared_ptr<ov::Model>& model) override;
    void validate_inputs(const ExpressionPtr& expr, const std::vector<TensorPtr>& inputs) override;
};

} // namespace lowered
} // namespace snippets
} // namespace ngraph
