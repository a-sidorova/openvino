//// Copyright (C) 2022 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "snippets/pass/lowered/softmax_decomposition.hpp"
//#include "snippets/pass/lowered/insert_loops_layout.hpp"
//#include "snippets/snippets_isa.hpp"
//#include "snippets/itt.hpp"
//#include <ngraph/pattern/op/wrap_type.hpp>
//#include "openvino/pass/pattern/matcher.hpp"
//#include "snippets/pass/lowered/insert_loops.hpp"
//
//namespace ngraph {
//namespace snippets {
//namespace pass {
//namespace lowered {
//using std::make_shared;
//SoftmaxDecomposition::SoftmaxDecomposition(size_t vector_size) : m_vector_size{vector_size} {}
//
//bool SoftmaxDecomposition::run(LoweredExprIR& linear_ir) {
//    OV_ITT_SCOPED_TASK(itt::domains::SnippetsTransform, "Snippets::SoftmaxDecompositionLowered")
//    auto match_softmax = ngraph::pattern::wrap_type<opset1::Softmax>();
//    auto matcher = std::make_shared<pattern::Matcher>(match_softmax, "SoftmaxDecompositionLowered");
//    bool modified = false;
//    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
//        const auto& op = (*expr_it)->get_node();
//        if (matcher->match(op)) {
//            const auto& pm = matcher->get_pattern_map();
//            const auto load_node = pm.at(match_softmax);
//            const auto softmax_expr = *expr_it;
//            const auto input_tds = softmax_expr->get_inputs();
//            const auto output_tds = expr_it->get()->get_outputs();
//            const auto tensor_out = output_tds.front()->get_tensor();
//            const auto subtensor_in = input_tds.front()->get_subtensor();
//
//            expr_it = linear_ir.erase(expr_it);   // Remove Softmax
//            auto outer_loop_end_pos = expr_it;
//            auto outer_loop_begin_pos = std::prev(expr_it);
//
//            const size_t buffer_allocation_rank = 2;
//            // We need an iterator to the inserted element
//            auto push_node = [&linear_ir, &expr_it](const std::shared_ptr<Node>& n) {
//                return std::make_pair(linear_ir.insert(expr_it, n), n);
//            };
//            std::vector<std::pair<LoweredExprIR::exprIt, LoweredExprIR::exprIt>> loop_begin_end_offsets;
//            // Note: VectorBuffer is a special case, since it should go before the initial Load. So we handle it separately
//            const auto& vector_buffer_max = push_node(make_shared<op::VectorBuffer>());
//
//            // Max loop
//            const auto& load_max_node = std::make_shared<op::Load>(load_node->get_input_source_output(0), m_vector_size);
//            auto loop_begin_offset = linear_ir.insert(expr_it, make_shared<LoweredExpr>(load_max_node, input_tds));
//            const auto& max = push_node(make_shared<ov::op::v1::Maximum>(load_max_node, vector_buffer_max.second));
//
//            const auto horizon_max = push_node(make_shared<op::HorizonMax>(max.second));
//            // Note: loopEnd will be inserted before HorizonMax
//            loop_begin_end_offsets.emplace_back(loop_begin_offset, horizon_max.first);
//            const auto broadcast_horizon_max = push_node(make_shared<op::BroadcastMove>(horizon_max.second,
//                                                                                           horizon_max.second->get_input_partial_shape(0)));
//            const auto vector_buffer_sum = push_node(make_shared<op::VectorBuffer>());
//
//            // Note: A Parameter can currently be connected only to one memory access child (usually Load). This is needed
//            // for upstream layout propagation. Here we insert op::Nop to indicate that layout from this Load should not
//            // be propagated to a parent Parameter.
//            const auto& load_sub_node = std::make_shared<op::Load>(load_node->get_input_source_output(0), m_vector_size);
//            loop_begin_offset = linear_ir.insert(expr_it, make_shared<LoweredExpr>(load_sub_node, input_tds));
//            const auto sub = push_node(make_shared<ov::op::v1::Subtract>(load_sub_node, broadcast_horizon_max.second));
//            const auto exp = push_node(make_shared<ov::op::v0::Exp>(sub.second));
//            const auto sum = push_node(make_shared<ov::op::v1::Add>(exp.second, vector_buffer_sum.second));
//            const auto store_exp = push_node(make_shared<op::Store>(exp.second, m_vector_size));
//            //const auto loop_end_sum = push_node(make_shared<op::LoopEnd>());
//
//            const auto horizon_sum = push_node(make_shared<op::HorizonSum>(sum.second));
//            loop_begin_end_offsets.emplace_back(loop_begin_offset, horizon_sum.first);
//            // Divide is expensive operation, so we decompose it into 1 / x * y, where 1 / x is executed outside loop
//            const auto pow = push_node(make_shared<op::PowerStatic>(horizon_sum.second, -1.));
//            const auto broadcast_pow = push_node(make_shared<op::BroadcastMove>(pow.second, horizon_sum.second->get_input_partial_shape(0)));
//            const auto buffer_exp = push_node(make_shared<op::Buffer>(store_exp.second, -1));
//
//            const auto load_div = push_node(make_shared<op::Load>(buffer_exp.second, m_vector_size));
//            loop_begin_offset = load_div.first;
//            const auto mul = push_node(make_shared<ov::op::v1::Multiply>(load_div.second, broadcast_pow.second));
//            const auto store_div_node = make_shared<op::Store>(mul.second, m_vector_size);
//            linear_ir.insert(expr_it, make_shared<LoweredExpr>(store_div_node, mul.first->get()->get_outputs(), output_tds));
//            loop_begin_end_offsets.emplace_back(loop_begin_offset, expr_it);
//
//            /* =========================================== */
//
//            /* ============= Runtime Info ================ */
//
//            // For tail loop we should fill input of Max by float min and
//            // input of Sum by zero to avoid math incorrect calculations
//            max.second->input(0).get_rt_info()["set_fill"] = uint32_t(0xff7fffff);
//            sum.second->input(0).get_rt_info()["set_fill"] = uint32_t(0x00000000);
//            size_t m_vector_size = 16;
//            for (const auto& begin_end : loop_begin_end_offsets) {
//                InsertLoops::insertion(linear_ir, begin_end.first, begin_end.second, 1, m_vector_size);
//            }
//
//            InsertLoops::insert_one_loop(linear_ir, std::next(outer_loop_begin_pos), outer_loop_end_pos,
//                                         *(tensor_out.rbegin() + 1),
//                                         1 < subtensor_in.size() ? *(subtensor_in.rbegin() + 1) : 1lu);
//            modified = true;
//        }
//    }
//    linear_ir.serialize("/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.xml",
//                        "/home/a-sidorova/projects/lin_ir/openvino/graphs/lin.bin");
//    return modified;
//}
//
//} // namespace lowered
//} // namespace pass
//} // namespace snippets
//} // namespace ngraph
//
