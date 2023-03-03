// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/insert_load_store.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

InsertLoadStore::InsertLoadStore(size_t vector_size) : m_vector_size(vector_size) {}

bool InsertLoadStore::insert_load(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt in_expr_it) {
    bool modified = false;
    const auto& td = (*in_expr_it)->get_outputs().front();
    const auto consumers = linear_ir.get_exprs_by_input(td);
    for (const auto& child_expr : consumers) {
        if (ov::is_type<op::Brgemm>(child_expr->get_node()) ||
            ov::is_type<op::Load>(child_expr->get_node()) ||
            ov::is_type<opset1::Result>(child_expr->get_node()))
            continue;

        const auto consumer_td = child_expr->get_outputs().front();
        const auto load_td = std::make_shared<TensorDescriptor>(td->get_tensor(),
                                                                td->get_subtensor(),
                                                                td->get_layout());
        // Port is 0 because Load can be inserted only on Parameter and Buffer output - they have only one out port
        const auto load = std::make_shared<op::Load>((*in_expr_it)->get_node()->output(0), m_vector_size);
        const auto load_outs = std::vector<TensorDescriptorPtr>{ load_td };
        const auto param_outs = std::vector<TensorDescriptorPtr>{ td };
        linear_ir.insert(std::find(linear_ir.begin(), linear_ir.end(), child_expr),
                         std::make_shared<LoweredExpr>(load, param_outs, load_outs));
        linear_ir.replace_input(child_expr, td, load_td);
        modified = true;
    }
    return modified;
}

bool InsertLoadStore::insert_store(LoweredExprIR& linear_ir, LoweredExprIR::constExprIt out_expr_it) {
    const auto& td = (*out_expr_it)->get_inputs().front();
    const auto parent_expr = linear_ir.get_expr_by_output(td);
    const auto parent = parent_expr->get_node();
    const auto port = parent_expr->get_output_port(td);

    if (ov::is_type<op::Store>(parent) ||
        ov::is_type<op::Buffer>(parent) ||
        ov::is_type<op::Brgemm>(parent) ||
        ov::is_type<opset1::Constant>(parent) ||
        ov::is_type<opset1::Parameter>(parent))
        return false;

    const auto store_td = std::make_shared<TensorDescriptor>(td->get_tensor(),
                                                             td->get_subtensor(),
                                                             td->get_layout());
    auto store = std::make_shared<op::Store>(parent->output(port), m_vector_size);
    std::vector<TensorDescriptorPtr> parent_outs { td };
    std::vector<TensorDescriptorPtr> store_outs { store_td };
    linear_ir.insert(std::next(std::find(linear_ir.begin(), linear_ir.end(), parent_expr)),
                     std::make_shared<LoweredExpr>(store, parent_outs, store_outs));
    linear_ir.replace_input((*out_expr_it), td, store_td);
    return true;
}

bool InsertLoadStore::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::InsertLoadStore")

    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto &op = expr_it->get()->get_node();
        if (ov::is_type<opset1::Parameter>(op) || ov::is_type<op::Buffer>(op)) {
            modified |= insert_load(linear_ir, expr_it);
        }
        if (ov::is_type<opset1::Result>(op) || ov::is_type<op::Buffer>(op)) {
            modified |= insert_store(linear_ir, expr_it);
        }
    }
    linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
                        "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
    return modified;
}



} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

