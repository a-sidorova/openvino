// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/runtime_config.hpp"

#include "snippets/op/memory_access.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {

bool RuntimeConfig::get_loop_desc(size_t loop_id, LoopDescriptor::Type type, LoopDescriptor& desc, size_t& index) const {
    OPENVINO_ASSERT(loops.count(loop_id) > 0, "LoopID has not been found!");
    index = 0;
    for (const auto& p : loops) {
        const auto& id = p.first;
        const auto& loop_descriptors = p.second;
        if (id == loop_id) {
            const auto desc_it = std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                                      [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
            if (desc_it != loop_descriptors.cend()) {
                desc = *desc_it;
                index += std::distance(loop_descriptors.cbegin(), desc_it);
                return true;
            }
            return false;
        }
        index += loop_descriptors.size();
    }
    return false;
}

bool RuntimeConfig::get_loop_desc(size_t loop_id, LoopDescriptor::Type type, LoopDescriptor& desc) const {
    OPENVINO_ASSERT(loops.count(loop_id) > 0, "LoopId has not been found!");
    const auto& loop_descriptors = loops.at(loop_id);
    const auto desc_it = std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                                      [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
    if (desc_it != loop_descriptors.cend()) {
        desc = *desc_it;
        return true;
    }
    return false;
}

void RuntimeConfig::update(const lowered::LinearIR& linear_ir) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    init_loop_descriptors(loop_manager);
    optimize_single_evaluation();
    init_data_offsets(linear_ir);
}

void RuntimeConfig::init_loop_descriptors(const lowered::LinearIR::LoopManagerPtr& loop_manager) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::init_loop_descriptors")
    loops.clear();
    const auto& loop_map = loop_manager->get_map();
    for (const auto& loop_pair : loop_map) {
        const auto loop_id = loop_pair.first;
        const auto loop_info = loop_pair.second;

        OPENVINO_ASSERT(loops.count(loop_id) == 0, "Loop is already in RuntimeConfig");
        loops[loop_id] = {};

        LoopDescriptor vector_loop_desc, tail_loop_desc;
        const auto vector_status = get_vector_loop_descriptor(loop_info, vector_loop_desc);
        const auto tail_status = get_tail_loop_descriptor(loop_info, vector_loop_desc, tail_loop_desc);
        if (vector_status) {
            loops[loop_id].push_back(vector_loop_desc);
        }
        if (tail_status) {
            loops[loop_id].push_back(tail_loop_desc);
            // Inner splited Loop update
            init_inner_splited_tail_loop_descriptors(loop_manager, loop_info, tail_loop_desc, loop_id, vector_status);
        }
    }
}

void RuntimeConfig::optimize_single_evaluation() {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::optimize_single_evaluation")
    for (auto& p : loops) {
        for (auto& loop_descriptor : p.second) {
            if (loop_descriptor.work_amount >= 2 * loop_descriptor.increment)
                continue;

            for (size_t i = 0; i < loop_descriptor.finalization_offsets.size(); i++) {
                loop_descriptor.finalization_offsets[i] += loop_descriptor.ptr_increments[i];
                loop_descriptor.ptr_increments[i] = 0;
            }
        }
    }
}

bool RuntimeConfig::get_vector_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, LoopDescriptor& vector_loop_desc) {
    if (!is_vector_loop_needed(loop_info))
        return false;

    const auto loop_ports = loop_info->get_all_ports();
    vector_loop_desc.work_amount = loop_info->work_amount;
    vector_loop_desc.increment = loop_info->increment;
    vector_loop_desc.ptr_increments.resize(loop_ports.size());
    vector_loop_desc.finalization_offsets.resize(loop_ports.size());
    vector_loop_desc.type = LoopDescriptor::Type::Vector;
    for (size_t i = 0; i < loop_ports.size(); ++i) {
        const auto& loop_port = loop_ports[i];
        vector_loop_desc.ptr_increments[i] = vector_loop_desc.increment * loop_port.ptr_increment * loop_port.data_size;
        vector_loop_desc.finalization_offsets[i] = loop_port.finalization_offset * loop_port.data_size;
    }
    return true;
}

bool RuntimeConfig::get_tail_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                                             LoopDescriptor& vector_loop_desc, LoopDescriptor& tail_loop_desc) {
    if (!is_tail_loop_needed(loop_info))
        return false;

    const auto loop_ports = loop_info->get_all_ports();
    const auto vector_needed = is_vector_loop_needed(loop_info);

    tail_loop_desc.work_amount = loop_info->work_amount % loop_info->increment;
    tail_loop_desc.increment = tail_loop_desc.work_amount;
    tail_loop_desc.ptr_increments.resize(loop_ports.size());
    tail_loop_desc.finalization_offsets.resize(loop_ports.size());
    tail_loop_desc.type = LoopDescriptor::Type::Tile;
    for (size_t i = 0; i < loop_ports.size(); ++i) {
        const auto& loop_port = loop_ports[i];
        tail_loop_desc.ptr_increments[i] = tail_loop_desc.increment * loop_port.ptr_increment * loop_port.data_size;
        if (!vector_needed)
            tail_loop_desc.finalization_offsets[i] = loop_port.finalization_offset * loop_port.data_size;
    }
    if (vector_needed) {
        tail_loop_desc.finalization_offsets = vector_loop_desc.finalization_offsets;
        std::fill(vector_loop_desc.finalization_offsets.begin(), vector_loop_desc.finalization_offsets.end(), 0);
    }
    return true;
}

void RuntimeConfig::init_inner_splited_tail_loop_descriptors(const LinearIR::LoopManagerPtr& loop_manager,
                                                            const LinearIR::LoopManager::LoopInfoPtr& outer_splited_loop_info,
                                                            const LoopDescriptor& outer_splited_tail_loop_desc,
                                                            size_t outer_loop_id, bool is_outer_vector_loop_needed) {
    if (!outer_splited_loop_info->outer_splited_loop)
        return;

    const auto tail_size = outer_splited_tail_loop_desc.work_amount;
    const auto outer_dim_idx = outer_splited_loop_info->dim_idx;
    const auto& loop_map = loop_manager->get_map();
    // go through all loops in loop manager to find inner loops by port loop IDs
    for (const auto& p : loop_map) {
        const auto loop_id = p.first;
        const auto loop_info = p.second;
        // skip the current loop
        if (loop_id == outer_loop_id)
            continue;

        // check if the target outer splited loop is really outer loop of the analyzed loop using loop IDs of ports
        OPENVINO_ASSERT(!loop_info->entry_points.empty(), "Each Loop must have one entry port at least!");
        const auto loop_port = loop_info->entry_points.front();
        const auto outer_loop_ids = LinearIR::LoopManager::get_outer_expr_loops(loop_port.expr_port->get_expr(), loop_id);
        if (std::find(outer_loop_ids.cbegin(), outer_loop_ids.cend(), outer_loop_id) == outer_loop_ids.cend())
            continue;

        // check if the target outer splited loop and the analyzed inner loop have the same dim_index
        const auto inner_dim_idx = loop_info->dim_idx;
        if (inner_dim_idx != outer_dim_idx)
            continue;

        LoopDescriptor splited_vector_loop_desc, splited_tail_loop_desc;
        OPENVINO_ASSERT(loops.count(loop_id) > 0 && loops[loop_id].size() == 1, "Splited inner Loop should be already inited!");
        OPENVINO_ASSERT(get_loop_desc(loop_id, LoopDescriptor::Type::Vector, splited_vector_loop_desc), "Splited inner Loop should be already inited!");
        splited_tail_loop_desc.work_amount = tail_size;
        splited_tail_loop_desc.increment = std::min(splited_vector_loop_desc.increment, tail_size);
        splited_tail_loop_desc.ptr_increments = splited_vector_loop_desc.ptr_increments;
        splited_tail_loop_desc.finalization_offsets = splited_vector_loop_desc.finalization_offsets;
        splited_tail_loop_desc.type = LoopDescriptor::Type::SplitedTile;
        // rescale offsets
        for (auto& offset : splited_tail_loop_desc.finalization_offsets) {
            offset = offset / static_cast<int64_t>(splited_vector_loop_desc.work_amount) * static_cast<int64_t>(splited_tail_loop_desc.work_amount);
        }
        loops[loop_id].push_back(splited_tail_loop_desc);
        // If outer splited loop doesn't have Vector Loop, inner splited loop shouldn't have the Vector Loop as well
        if (!is_outer_vector_loop_needed) {
            loops[loop_id].erase(loops[loop_id].begin()); // since we check that there is only one descriptor above
        }
    }
}

void RuntimeConfig::init_data_offsets(const LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::init_data_offsets")

    const auto tensor_rank = linear_ir.get_config().m_tensor_rank;
    const auto& io_exprs = linear_ir.get_IO_ops();
    data_offsets.resize(io_exprs.size());

    size_t idx = 0;
    for (const auto& expr : io_exprs) {
        lowered::PortDescriptorPtr desc = nullptr;
        element::Type etype;
        switch (expr->get_type()) {
            case lowered::IOExpression::io_type::INPUT: {
                // Note that here we consider only the first child (which is usually load),
                // but often there is another child - LoopEnd
                auto consumer_inputs = expr->get_output_port_connector(0)->get_consumers();
                const auto& first_consumer = consumer_inputs.begin()->get_expr();
                // If there is a RankNormalization op after a parameter - we should skip it
                if (is_type<snippets::op::RankNormalization>(first_consumer->get_node()))
                    consumer_inputs = first_consumer->get_output_port_connector(0)->get_consumers();
                for (const auto& child_input : consumer_inputs) {
                    if (ov::as_type_ptr<snippets::op::MemoryAccess>(child_input.get_expr()->get_node())) {
                        desc = child_input.get_descriptor_ptr();
                        break;
                    }
                }
                etype = expr->get_node()->get_output_element_type(0);
                break;
            }
            case lowered::IOExpression::io_type::OUTPUT: {
                desc = expr->get_input_port_connector(0)->get_source().get_descriptor_ptr();
                etype = expr->get_node()->get_input_element_type(0);
                break;
            } default : {
                OPENVINO_THROW("Detected unsupported io_type");
            }
        }
        offset_calculation(desc, etype.size(), expr->get_type() == lowered::IOExpression::io_type::INPUT, tensor_rank, data_offsets[idx++]);
    }
}

void RuntimeConfig::offset_calculation(const lowered::PortDescriptorPtr& desc, size_t data_size, bool is_input, size_t rank, std::vector<int64_t>& offsets) {
    // offsets represent distance between consecutive elements of corresponding dimension.
    // If a dim size == 1, then the next dim starts immediately and the stride is 0
    // case 1:
    //    shape:         s0,    s1, s2, s3
    //    offsets: s1*s2*s3, s2*s3, s3,  1
    // case 2:
    //    shape:      s0, s1, s2 == 1, s3
    //    offsets: s1*s3, s3,       0,  1
    const auto& shape = desc->get_shape();
    const auto& layout = desc->get_layout();

    offsets.resize(rank);
    std::fill(offsets.begin(), offsets.end(), 0);
    OPENVINO_ASSERT(rank >= shape.size(), "Incorrect tensor rank!");
    const auto idx_stride = rank - shape.size();
    int64_t dim_step = static_cast<int64_t>(data_size);
    offsets[offsets.size() - 1] = dim_step;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        dim_step *= static_cast<int64_t>(shape[i + 1]);
        offsets[i + idx_stride] = shape[i] != 1 ? dim_step : 0;
    }
    if (!layout.empty()) {
        std::vector<int64_t> reordered_offsets(offsets.size());
        for (size_t i = 0; i < layout.size(); i++) {
            const auto& src_idx = is_input ? layout[i] : i;
            const auto& dst_idx = is_input ? i : layout[i];
            reordered_offsets[idx_stride + dst_idx] = offsets[idx_stride + src_idx];
        }
        offsets = std::move(reordered_offsets);
    }
}

}// namespace snippets
}// namespace ov
