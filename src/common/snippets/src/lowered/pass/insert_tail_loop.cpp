// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_tail_loop.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void InsertTailLoop::update_loop_id_mapping(const std::shared_ptr<op::LoopEnd> loop_end, const size_t new_id, const size_t old_id) {
    loop_end->set_id(new_id);
    m_loop_ids_mapping[new_id] = old_id;
}

void InsertTailLoop::propagate_updated_subtensor_through_loop(const LinearIR& linear_ir,
                                                              const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                                                              LinearIR::container::const_iterator begin,
                                                              LinearIR::container::const_iterator end,
                                                              const size_t new_dim_value) {
    std::map<lowered::PortDescriptorPtr, snippets::VectorDims> original_shapes;
    // First step: set new dim value to the corresponding entry_points' dimensions
    if (new_dim_value != existing_subtensor_value) {
        for (const auto& port : loop_info->get_entry_points()) {
            if (port.is_incremented) {
                const auto& expr = port.expr_port->get_expr();
                const auto node = expr->get_node();
                auto desc = port.expr_port->get_descriptor_ptr();
                auto subtensor = desc->get_subtensor();
                if (port.dim_idx < subtensor.size()) {
                    *(subtensor.rbegin() + port.dim_idx) = new_dim_value;
                    desc->set_subtensor(subtensor);
                }

                const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();
                const auto& shape = parent_desc->get_shape();
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = shape;
                }
                const auto& layout = desc->get_layout();
                auto new_shape = shape;
                new_shape[*(layout.rbegin() + port.dim_idx)] = new_dim_value;
                parent_desc->set_shape(new_shape);
            }
        }
    }

    auto update_only_dim_idx_with_subtensor_value = [&](const LinearIR::LoopManager::LoopPort& port) {
        if (port.is_incremented) {
            auto desc = port.expr_port->get_descriptor_ptr();
            const auto expr = port.expr_port->get_expr();
            const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();

            const auto& shape = parent_desc->get_shape();
            const auto& desc_subtensor = desc->get_subtensor();
            if (port.dim_idx < desc_subtensor.size()) {
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = shape;
                }
                const auto& layout = desc->get_layout();
                auto new_shape = shape;
                new_shape[*(layout.rbegin() + port.dim_idx)] = *(desc_subtensor.rbegin() + port.dim_idx);
                parent_desc->set_shape(new_shape);
            }
        }
    };

    auto update_subtensors = [](const std::vector<PortDescriptorPtr>& descs, bool is_input) {
        for (const auto& desc : descs) {
            const auto& subtensor = desc->get_subtensor();
            if (!subtensor.empty()) {
                auto planar_dims = is_input ? snippets::utils::get_planar_vdims(desc->get_shape(), desc->get_layout())
                                            : snippets::utils::get_preordered_vdims(desc->get_shape(), desc->get_layout());
                const size_t subtensor_start = planar_dims.size() - subtensor.size();
                VectorDims new_subtensor(planar_dims.begin() + subtensor_start, planar_dims.end());
                for (size_t i = 0; i < new_subtensor.size(); ++i) {
                    new_subtensor[i] = std::min(new_subtensor[i], subtensor[i]);
                }
                desc->set_subtensor(new_subtensor);
            }
        }
    };

    auto shape_inference_end_it = end;
    const bool loop_by_last_dim = loop_info->get_dim_idx() == 0;
    // Subtensors are updated using shape inference infrastructure:
    // For inner loops propagation function is called recursively
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        if (ov::is_type<snippets::op::LoopEnd>(expr->get_node()))
            continue;
        if (auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node())) {
            const auto loop_end = loop_begin->get_loop_end();
            const auto inner_loop_info = linear_ir.get_loop_manager()->get_loop_info(m_loop_ids_mapping.at(loop_end->get_id()));
            const auto inner_begin = std::next(expr_it);
            const auto inner_end = linear_ir.find(linear_ir.get_expr_by_node(loop_end));

            // The corresponding shapes of inner loops entry points must be updated using existing subtensor values
            if (new_dim_value == existing_subtensor_value) {
                for (const auto& port : loop_info->get_entry_points())
                    update_only_dim_idx_with_subtensor_value(port);
            }
            propagate_updated_subtensor_through_loop(linear_ir, inner_loop_info, inner_begin, inner_end);
            expr_it = inner_end;
            continue;
        }
        if ((ov::is_type<snippets::op::BroadcastMove>(expr_it->get()->get_node()) ||
            ov::is_type<snippets::op::BroadcastLoad>(expr_it->get()->get_node())) &&
            loop_by_last_dim) {
            // WA: we have to break subtensor propagation if we try to propagate new last dim through Broadcast nodes
            // which broadcast last dim in original dimension value anyway
            // This workaround might be avoided if blocked shape are used for tail size propagation
            shape_inference_end_it = expr_it;
            break;
        }
        expr->updateShapes();
        update_subtensors(expr->get_input_port_descriptors(), true);
        update_subtensors(expr->get_output_port_descriptors(), false);
    }

    // After subtensor propagation, the original shapes must be restored
    for (const auto& elem : original_shapes)
        elem.first->set_shape(elem.second);
    for (auto expr_it = begin; expr_it != shape_inference_end_it; expr_it++)
        (*expr_it)->updateShapes();
}

LinearIR::container InsertTailLoop::copy_loop(const LinearIR& linear_ir, LinearIR::constExprIt loop_begin_pos, LinearIR::constExprIt loop_end_pos,
                                              size_t loop_id) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    ExressionMap expression_map;
    const auto& loop_copy_range = LinearIR::deep_copy_range(loop_begin_pos, std::next(loop_end_pos), expression_map);

    const auto original_loop_info = loop_manager->get_loop_info(loop_id);
    std::vector<LinearIR::LoopManager::LoopPort> new_entry_points, new_exit_points;
    // Clone loop ports from original loop info to new loop info
    for (const auto& entry : original_loop_info->get_entry_points())
        new_entry_points.push_back(*entry.clone_with_new_expr(expression_map[entry.expr_port->get_expr().get()]));
    for (const auto& exit : original_loop_info->get_exit_points())
        new_exit_points.push_back(*exit.clone_with_new_expr(expression_map[exit.expr_port->get_expr().get()]));

    for (const auto& elem : expression_map) {
        const auto expr = elem.first->shared_from_this();
        const auto& new_expr = elem.second;
        // Loop begin/end ops can't be loop ports
        if (ov::is_type<op::LoopBase>(expr->get_node()))
            continue;
        // Update loop info of all outer loops with new loop ports
        const auto outer_loop_ids = LinearIR::LoopManager::get_outer_expr_loops(expr, loop_id);
        for (size_t i = 0; i < expr->get_input_count(); ++i)
            loop_manager->update_loops_port(outer_loop_ids, expr->get_input_port(i), {expr->get_input_port(i), new_expr->get_input_port(i)}, true);
        for (size_t i = 0; i < expr->get_output_count(); ++i)
            loop_manager->update_loops_port(outer_loop_ids, expr->get_output_port(i), {expr->get_output_port(i), new_expr->get_output_port(i)}, false);
    }

    const auto new_loop_begin_pos = loop_copy_range.begin();
    const auto new_loop_end_pos = loop_copy_range.end();
    const auto new_id = loop_manager->replace_with_new_loop(linear_ir,
                                                            std::next(new_loop_begin_pos),
                                                            std::prev(new_loop_end_pos),
                                                            original_loop_info->get_work_amount(),
                                                            original_loop_info->get_increment(),
                                                            new_entry_points,
                                                            new_exit_points,
                                                            loop_id);
    loop_manager->get_loop_info(new_id)->set_first_iter_handler(original_loop_info->get_first_iter_handler());
    loop_manager->get_loop_info(new_id)->set_outer_splited_loop(original_loop_info->get_outer_splited_loop());
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(std::prev(new_loop_end_pos)->get()->get_node());
    OPENVINO_ASSERT(loop_end, "Cloned Loop does not contain LoopEnd op at the expected place.");
    loop_end->set_id(new_id);
    return loop_copy_range;
}

bool InsertTailLoop::init_main_loop(size_t loop_id, const std::shared_ptr<op::LoopEnd>& loop_end, RuntimeConfig::LoopDescriptor::Type type) {
    size_t updated_loop_id;
    RuntimeConfig::LoopDescriptor loop_desc;
    if (m_runtime_config.get_loop_desc(loop_id, type, loop_desc, updated_loop_id)) {
        loop_end->update(loop_desc);
        update_loop_id_mapping(loop_end, updated_loop_id, loop_end->get_id());
        return true;
    }
    return false;
}

bool InsertTailLoop::create_first_iter_loop(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                            size_t original_loop_id, const std::shared_ptr<op::LoopEnd>& loop_end) {
    size_t updated_loop_id;
    RuntimeConfig::LoopDescriptor loop_desc;
    if (m_runtime_config.get_loop_desc(original_loop_id, RuntimeConfig::LoopDescriptor::Type::First, loop_desc, updated_loop_id)) {
        std::shared_ptr<op::LoopEnd> first_iter_loop_end = loop_end;
        LinearIR::constExprIt first_iter_loop_end_it = end;
        // Need to copy body if there are other specific sup-loops
        // Otherwise we should update the current body
        const bool need_copy_body = m_runtime_config.get_loops().at(original_loop_id).size() > 1;
        if (need_copy_body) {
            const auto new_loop_range = copy_loop(linear_ir, begin, end, loop_end->get_id());
            linear_ir.insert(begin, new_loop_range.begin(), new_loop_range.end());
            first_iter_loop_end_it = std::prev(begin);
            first_iter_loop_end = ov::as_type_ptr<op::LoopEnd>(first_iter_loop_end_it->get()->get_node());
            OPENVINO_ASSERT(first_iter_loop_end != nullptr, "Cloned Loop does not contain LoopEnd op at the expected place.");
        }
        first_iter_loop_end->update(loop_desc);

        const auto& loop_manager = linear_ir.get_loop_manager();
        const auto loop_id = first_iter_loop_end->get_id();
        const auto& first_iter_handler = loop_manager->get_loop_info(loop_id)->get_first_iter_handler();
        OPENVINO_ASSERT(first_iter_handler != nullptr, "First Iter handler is missed!");
        first_iter_handler(linear_ir, first_iter_loop_end_it);

        update_loop_id_mapping(first_iter_loop_end, updated_loop_id, loop_id);
        return true;
    }
    return false;
}

bool InsertTailLoop::create_tail_loop(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                      size_t original_loop_id, std::shared_ptr<op::LoopEnd>& loop_end) {
    size_t updated_loop_id;
    RuntimeConfig::LoopDescriptor loop_desc;
    if (m_runtime_config.get_loop_desc(original_loop_id, RuntimeConfig::LoopDescriptor::Type::Last, loop_desc, updated_loop_id)) {
        const auto tail_loop_end = loop_end;
        // Need to copy body if there is main loop
        // Otherwise we should update the current body
        const bool need_copy_body = m_runtime_config.contains(original_loop_id, RuntimeConfig::LoopDescriptor::Type::Main);
        if (need_copy_body) {
            const auto new_loop_range = copy_loop(linear_ir, begin, end, original_loop_id);
            const auto vector_loop_end = ov::as_type_ptr<op::LoopEnd>(std::prev(new_loop_range.end())->get()->get_node());
            OPENVINO_ASSERT(vector_loop_end != nullptr, "Cloned Loop does not contain LoopEnd op at the expected place.");
            // Notes:
            // - new loop body is inserted before the original loop
            //   So new loop becomes a main vector loop, the original loop becomes tail loop
            //   This is done in such way to have original ops from the main body at the end:
            //   this allows us to conveniently interact with outer loops in further passes
            // - vector loop is already inited by descriptor
            linear_ir.insert(begin, new_loop_range.begin(), new_loop_range.end());
            // Since we copy main loop, we have to update loop_end
            loop_end = vector_loop_end;
        }
        tail_loop_end->update(loop_desc);

        const auto& loop_manager = linear_ir.get_loop_manager();
        const auto loop_info = loop_manager->get_loop_info(original_loop_id);
        update_loop_id_mapping(tail_loop_end, updated_loop_id, original_loop_id);

        // We have to check the loop body for any nested loops that work on the same dimension
        // and rescale their work_amount and increment accordingly
        if (loop_info->get_outer_splited_loop()) {
            const auto current_dim_idx = loop_info->get_dim_idx();
            OPENVINO_ASSERT(current_dim_idx != LinearIR::LoopManager::LoopInfo::UNDEFINED_DIM_IDX,
                                "Outer splitted loop unexpectedly iterates by several dimension indices");
            size_t updated_loop_id;
            for (auto it = std::next(begin); it != end; ++it) {
                const auto& expr = *it;
                const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
                if (!inner_loop_end)
                    continue;
                // Use Loop ID mapping since these Loops are already inited
                const auto inner_loop_info = loop_manager->get_loop_info(m_loop_ids_mapping.at(inner_loop_end->get_id()));
                const auto inner_dim_idx = inner_loop_info->get_dim_idx();
                if (inner_dim_idx != current_dim_idx)
                    continue;

                RuntimeConfig::LoopDescriptor splited_tail_loop_desc;
                OPENVINO_ASSERT(m_runtime_config.get_loop_desc(m_loop_ids_mapping.at(inner_loop_end->get_id()),
                                                               RuntimeConfig::LoopDescriptor::Type::SplitedLast,
                                                               splited_tail_loop_desc, updated_loop_id),
                                "Splited inner Loop has not been found!");
                inner_loop_end->update(splited_tail_loop_desc);
                update_loop_id_mapping(inner_loop_end, updated_loop_id, m_loop_ids_mapping.at(inner_loop_end->get_id()));

                const auto inner_loop_begin_it = std::find(begin, it, linear_ir.get_expr_by_node(inner_loop_end->get_loop_begin()));
                const auto inner_loop_end_it = std::next(end);
                OPENVINO_ASSERT(inner_loop_begin_it != it, "LoopBegin has not been found!");
                tail_transformations(linear_ir, inner_loop_begin_it, inner_loop_end_it, loop_desc.increment);
            }
        }
        tail_transformations(linear_ir, begin, end, loop_desc.increment);
        propagate_updated_subtensor_through_loop(linear_ir, loop_info, std::next(begin), end, loop_desc.increment);
        return true;
    }
    return false;
}

void InsertTailLoop::tail_transformations(LinearIR& linear_ir,
                                          LinearIR::constExprIt tail_begin,
                                          LinearIR::constExprIt tail_end,
                                          const size_t tail_size) {
    const auto& config = linear_ir.get_config();
    auto insertFill = [tail_size](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
        std::shared_ptr<ov::Node> fill = nullptr;
        auto& rt = input.get_rt_info();
        auto fill_rt = rt.find("set_fill");
        if (fill_rt != rt.end()) {
            const auto fill_value = fill_rt->second.as<uint32_t>();
            fill = std::make_shared<ov::snippets::op::Fill>(input.get_source_output(), tail_size, fill_value);
            input.get_node()->set_argument(input.get_index(), fill);
        }
        return fill;
    };

    for (auto expr_it = std::next(tail_begin); expr_it != tail_end; expr_it++) {
        // Skip inner Loops
        const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(expr_it->get()->get_node());
        if (loop_begin) {
            expr_it = linear_ir.find(expr_it, tail_end, linear_ir.get_expr_by_node(loop_begin->get_loop_end()));
            continue;
        }
        // We should fill vector regs by float_min and zero to have
        // correct math calculations for ReduceMax and ReduceSum in scalar case.
        // Note: We find Maximum and Add ops because HorizonMax and HorizonSum are outside Loop,
        //       so they are missed in <tail>
        const auto& expr = *expr_it;
        const auto op = expr->get_node();
        if (config.m_need_fill_tail_register &&
            (ov::is_type<ov::op::v1::Maximum>(op) ||
             ov::is_type<ov::op::v1::Add>(op))) {
            for (size_t i = 0; i < op->inputs().size(); ++i) {
                if (auto fill = insertFill(op->input(i))) {
                    const auto& input = expr->get_input_port_connector(i);
                    const auto consumers = input->get_consumers();
                    // If there are several consumers, fill expression must be inserted before first of them
                    auto fst_consumer = std::min_element(consumers.cbegin(), consumers.cend(), [&](ExpressionPort lhs, ExpressionPort rhs) {
                        auto lhs_it = linear_ir.find(lhs.get_expr());
                        auto rhs_it = linear_ir.find(rhs.get_expr());
                        return std::distance(linear_ir.cbegin(), lhs_it) < std::distance(linear_ir.cbegin(), rhs_it);
                    });
                    const auto insert_pos = linear_ir.find(fst_consumer->get_expr());
                    auto fill_expr = linear_ir.create_expression(fill, {input});
                    linear_ir.insert(insert_pos, fill_expr);
                    linear_ir.replace_input(consumers, fill_expr->get_output_port_connector(0));
                    // in_reg == out_reg since we want to modify vector reg inplace
                    const auto reg = expr->get_input_port_descriptor(0)->get_reg();
                    fill_expr->get_input_port_descriptor(0)->set_reg(reg);
                    fill_expr->get_output_port_descriptor(0)->set_reg(reg);
                    fill_expr->set_loop_ids(expr->get_loop_ids());
                }
            }
        } else if (const auto memory_access = std::dynamic_pointer_cast<ov::snippets::op::MemoryAccess>(op)) {
            for (const auto p : memory_access->get_memory_access_input_ports()) {
                const auto port = p.first;
                if (memory_access->get_input_count(port) > 1) {
                    memory_access->set_input_count(tail_size, port);
                }
            }
            for (const auto p : memory_access->get_memory_access_output_ports()) {
                const auto port = p.first;
                if (memory_access->get_output_count(port) > 1) {
                    memory_access->set_output_count(tail_size, port);
                }
            }
        }
    }
}

bool InsertTailLoop::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::insertTailLoop")
    bool modified = false;

    const auto& loop_descriptors = m_runtime_config.get_loops();

    m_loop_ids_mapping.clear();
    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); ++expr_it) {
        const auto& expr = *expr_it;
        const auto node = expr->get_node();
        auto main_loop_end = ov::as_type_ptr<op::LoopEnd>(node);
        if (!main_loop_end)
            continue;

        const auto begin_it = linear_ir.find(linear_ir.get_expr_by_node(main_loop_end->get_loop_begin()));

        const auto loop_id = main_loop_end->get_id();
        OPENVINO_ASSERT(loop_descriptors.find(loop_id) != loop_descriptors.cend() && !loop_descriptors.at(loop_id).empty(),
                        "LoopDescriptors are missed for Loop with ID " + std::to_string(loop_id));

        // Attention: the order of loop creation and initialization is important!!!
        // Corner case with splited inner loop where there is only tile descriptor without vector loop descriptor
        const auto one_splitted_tail_loop_status = loop_descriptors.at(loop_id).size() == 1 &&
                                                   init_main_loop(loop_id, main_loop_end, RuntimeConfig::LoopDescriptor::Type::SplitedLast);
        const auto first_iter_status =
            create_first_iter_loop(linear_ir, begin_it, expr_it, loop_id, main_loop_end);
        const auto last_iter_status =
            create_tail_loop(linear_ir, begin_it, expr_it, loop_id, main_loop_end);
        const auto vector_loop_status =
            init_main_loop(loop_id, main_loop_end, RuntimeConfig::LoopDescriptor::Type::Main);
        OPENVINO_ASSERT(one_splitted_tail_loop_status || vector_loop_status || first_iter_status || last_iter_status, "The Loop has not been updated!");
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

