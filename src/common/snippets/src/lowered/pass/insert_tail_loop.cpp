// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_tail_loop.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

std::shared_ptr<op::LoopEnd> InsertTailLoop::create_tail_loop(LinearIR& linear_ir,
                                                              LinearIR::constExprIt vector_begin, LinearIR::constExprIt vector_end,
                                                              LinearIR::constExprIt& tail_begin, LinearIR::constExprIt& tail_end,
                                                              const std::shared_ptr<op::LoopEnd>& vector_loop_end,
                                                              bool is_vector_inserted,
                                                              const RuntimeConfig::LoopDescriptor& tail_loop_desc,
                                                              const std::map<size_t, size_t>& updated_loop_ids) {
    tail_begin = vector_begin, tail_end = vector_end;
    if (is_vector_inserted) {
        ExressionMap expression_map;
        auto vector_loop_deep_copy = LinearIR::deep_copy_range(vector_begin, vector_end, expression_map);
        tail_begin = linear_ir.insert(vector_end, vector_loop_deep_copy.cbegin(), vector_loop_deep_copy.cend());
    }

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& current_loop_info = loop_manager->get_loop_info(is_vector_inserted ? updated_loop_ids.at(vector_loop_end->get_id())
                                                                                   : vector_loop_end->get_id());
    if (current_loop_info->outer_splited_loop) {
        const auto current_dim_idx = current_loop_info->dim_idx;
        size_t updated_loop_id;
        for (auto it = std::next(tail_begin); it != std::prev(tail_end); ++it) {
            const auto& expr = *it;
            const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
            if (!inner_loop_end)
                continue;

            const auto loop_info = loop_manager->get_loop_info(updated_loop_ids.at(inner_loop_end->get_id()));
            if (loop_info->dim_idx != current_dim_idx)
                continue;

            RuntimeConfig::LoopDescriptor splited_tail_loop_desc;
            OPENVINO_ASSERT(m_runtime_config.get_loop_desc(updated_loop_ids.at(inner_loop_end->get_id()), RuntimeConfig::LoopDescriptor::Type::SplitedTile,
                                                           splited_tail_loop_desc, updated_loop_id),
                            "Splited inner Loop has not been found!");
            inner_loop_end->update(splited_tail_loop_desc);
            const auto inner_loop_begin_it = linear_ir.find(tail_begin, it, linear_ir.get_expr_by_node(inner_loop_end->get_loop_begin()));
            const auto inner_loop_end_it = std::next(tail_end);
            OPENVINO_ASSERT(inner_loop_begin_it != it, "LoopBegin has not been found!");
            tail_transformations(linear_ir, inner_loop_begin_it, inner_loop_end_it, tail_loop_desc.work_amount);
            inner_loop_end->set_id(updated_loop_id);
        }
    }

    tail_transformations(linear_ir, tail_begin, tail_end, tail_loop_desc.work_amount);
    const auto tail_loop_end = ov::as_type_ptr<op::LoopBegin>((*tail_begin)->get_node())->get_loop_end();
    tail_loop_end->update(tail_loop_desc);
    return tail_loop_end;
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
        auto op = (*expr_it)->get_node();
        if (config.m_need_fill_tail_register &&
            (ov::is_type<ov::op::v1::Maximum>(op) ||
             ov::is_type<ov::op::v1::Add>(op))) {
            for (size_t i = 0; i < op->inputs().size(); ++i) {
                if (auto fill = insertFill(op->input(i))) {
                    const auto& input = expr_it->get()->get_input_port_connector(i);
                    const auto consumers = input->get_consumers();
                    auto fill_expr = linear_ir.create_expression(fill, {input});
                    linear_ir.insert(expr_it, fill_expr);
                    linear_ir.replace_input(consumers, fill_expr->get_output_port_connector(0));
                    // in_reg == out_reg since we want to modify vector reg inplace
                    const auto reg = expr_it->get()->get_input_port_descriptor(0)->get_reg();
                    fill_expr->get_input_port_descriptor(0)->set_reg(reg);
                    fill_expr->get_output_port_descriptor(0)->set_reg(reg);
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
    const auto& loop_descriptors = m_runtime_config.get_loops();

    // [new id in RuntimeConfig::LoopDescriptor bounds] -> [old id in LoopManager bounds]
    std::map<size_t, size_t> updated_loop_ids;
    size_t updated_loop_id;

    auto update_loop_id = [&updated_loop_ids](const std::shared_ptr<op::LoopEnd> loop_end, const size_t new_id, const size_t old_id) {
        loop_end->set_id(new_id);
        updated_loop_ids[new_id] = old_id;
    };

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); ++expr_it) {
        const auto& expr = *expr_it;
        const auto node = expr->get_node();
        const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
        if (!loop_end)
            continue;

        const auto loop_id = loop_end->get_id();
        OPENVINO_ASSERT(loop_descriptors.find(loop_id) != loop_descriptors.cend() && !loop_descriptors.at(loop_id).empty(),
                        "LoopDescriptors are missed for Loop with ID " + std::to_string(loop_id));

        bool updated = false, need_vector_loop = false;
        RuntimeConfig::LoopDescriptor vector_loop_desc, tail_loop_desc;
        // Corner case with splited inner loop where there is only tile descriptor without vector loop descriptor
        if (loop_descriptors.at(loop_id).size() == 1) {
            RuntimeConfig::LoopDescriptor splited_loop_desc;
            if (m_runtime_config.get_loop_desc(loop_id, RuntimeConfig::LoopDescriptor::Type::SplitedTile, splited_loop_desc, updated_loop_id)) {
                update_loop_id(loop_end, updated_loop_id, loop_id);
                updated = true;
            }
        }

        if (m_runtime_config.get_loop_desc(loop_id, RuntimeConfig::LoopDescriptor::Type::Vector, vector_loop_desc, updated_loop_id)) {
            loop_end->update(vector_loop_desc);
            update_loop_id(loop_end, updated_loop_id, loop_id);
            need_vector_loop = true;
            updated = true;
        }

        if (m_runtime_config.get_loop_desc(loop_id, RuntimeConfig::LoopDescriptor::Type::Tile, tail_loop_desc, updated_loop_id)) {
            const auto loop_begin = loop_end->get_loop_begin();
            const auto begin_it = linear_ir.find(linear_ir.get_expr_by_node(loop_begin));
            LinearIR::constExprIt tail_begin, tail_end;
            const auto tail_loop_end = create_tail_loop(linear_ir, begin_it, std::next(expr_it), tail_begin, tail_end,
                                                        loop_end, need_vector_loop, tail_loop_desc, updated_loop_ids);
            // Skip new tail loop. Note: tail_end refs to the next expression after LoopEnd of tail
            expr_it = std::prev(tail_end);
            OPENVINO_ASSERT(tail_loop_end != nullptr, "After Tail insertion there should be LoopEnd op");
            update_loop_id(tail_loop_end, updated_loop_id, loop_id);
            updated = true;
        }
        OPENVINO_ASSERT(updated, "The Loop has not been updated!");
    }
    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

