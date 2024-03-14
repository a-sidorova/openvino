// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/runtime_configurator.hpp"

#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {

RuntimeConfigurator::RuntimeConfigurator() {
    m_desc_initializers[RuntimeConfig::LoopDescriptor::Type::First] = std::make_shared<RuntimeConfigurator::FirstLoopInitializer>();
    m_desc_initializers[RuntimeConfig::LoopDescriptor::Type::Main] = std::make_shared<RuntimeConfigurator::MainLoopInitializer>();
    m_desc_initializers[RuntimeConfig::LoopDescriptor::Type::Last] = std::make_shared<RuntimeConfigurator::LastLoopInitializer>();
}

const RuntimeConfig& RuntimeConfigurator::init(const lowered::LinearIR& linear_ir) {
    if (m_inited) {
        reset();
    }
    init_io_info(linear_ir);
    init_loop_descriptors(linear_ir.get_loop_manager());
    init_data_offsets();
    m_inited = true;
    return m_config;
}

const RuntimeConfig& RuntimeConfigurator::update(const lowered::LinearIR& linear_ir) {
    OPENVINO_ASSERT(m_inited, "Configurator must be already inited!");
    update_loop_descriptors(linear_ir.get_loop_manager());
    init_data_offsets();
    return m_config;
}

std::shared_ptr<RuntimeConfigurator> RuntimeConfigurator::clone(const lowered::LinearIR& linear_ir) {
    auto cloned = std::make_shared<RuntimeConfigurator>(*this);
    if (m_inited)
        cloned->init_io_info(linear_ir);
    return cloned;
}

void RuntimeConfigurator::reset() {
    m_inited = false;
    m_config.clear();
}

void RuntimeConfigurator::init_loop_descriptors(const lowered::LinearIR::LoopManagerPtr& loop_manager) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::init_loop_descriptors")

    const auto& loop_map = loop_manager->get_map();
    for (const auto& loop_pair : loop_map) {
        const auto loop_id = loop_pair.first;
        // make a copy to avoid original loop info corruption
        const auto loop_info = std::make_shared<LinearIR::LoopManager::LoopInfo>(*loop_pair.second);
        lowered::pass::InitLoops::init_loop_info(loop_info, true);

        OPENVINO_ASSERT(!utils::is_dynamic_value(loop_info->get_increment()), "Increment must be static value!");
        OPENVINO_ASSERT(!m_config.contains(loop_id),  "There should not be loop descriptors at first initialization");
        m_config.m_loops[loop_id] = {};

        for (const auto& p : m_desc_initializers) {
            const auto& type = p.first;
            const auto& intializer = p.second;
            if (intializer->is_needed(loop_info)) {
                auto desc_it = m_config.push_new_desc(loop_id, type);
                intializer->init_descriptor(loop_manager, loop_info, loop_id, *desc_it, m_config);
            }
        }
    }
}

void RuntimeConfigurator::update_loop_descriptors(const lowered::LinearIR::LoopManagerPtr& loop_manager) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::update_loop_descriptors")

    const auto& loop_map = loop_manager->get_map();
    for (const auto& loop_pair : loop_map) {
        const auto loop_id = loop_pair.first;
        // make a copy to avoid original loop info corruption
        const auto loop_info = std::make_shared<LinearIR::LoopManager::LoopInfo>(*loop_pair.second);
        lowered::pass::InitLoops::init_loop_info(loop_info, true);

        OPENVINO_ASSERT(!utils::is_dynamic_value(loop_info->get_increment()), "Increment must be static value!");
        OPENVINO_ASSERT(m_config.contains(loop_id), "There must be already inited loop descs");

        for (auto& loop_desc : m_config.m_loops.at(loop_id)) {
            const auto& initializer = m_desc_initializers.at(loop_desc.type);
            if (initializer->is_needed(loop_info)) {
                initializer->update_descriptor(loop_manager, loop_info, loop_id, loop_desc, m_config);
            } else {
                // To skip loop evaluation - set zero as work amount is enough
                loop_desc.work_amount = 0;
            }
        }
    }
}

void RuntimeConfigurator::init_io_info(const LinearIR& linear_ir) {
    const auto& io_exprs = linear_ir.get_IO_ops();
    m_io_num = io_exprs.size();
    m_config.m_data_offsets.resize(m_io_num);
    m_io_descs.resize(m_io_num);
    m_io_data_sizes.resize(m_io_num);
    m_in_num = 0;
    m_tensor_rank = linear_ir.get_config().m_tensor_rank;

    size_t current_rank = 0;
    size_t idx = 0;
    for (const auto& expr : io_exprs) {
        switch (expr->get_type()) {
            case lowered::IOExpression::io_type::INPUT: {
                // Note that here we consider only the first child (which is usually load),
                // but often there is another child - LoopEnd
                auto consumer_inputs = expr->get_output_port_connector(0)->get_consumers();
                const auto& first_consumer = consumer_inputs.begin()->get_expr();
                // If there is a RankNormalization op after a parameter - we should skip it
                if (is_type<snippets::op::RankNormalization>(first_consumer->get_node()))
                    consumer_inputs = first_consumer->get_output_port_connector(0)->get_consumers();
                // TODO: Add validation pass after control flow pipeline that all consumers have the same layout
                for (const auto& child_input : consumer_inputs) {
                    const auto ma = ov::as_type_ptr<snippets::op::MemoryAccess>(child_input.get_expr()->get_node());
                    if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                        m_io_descs[idx] = child_input.get_descriptor_ptr();
                    }
                }
                m_io_data_sizes[idx] = expr->get_node()->get_output_element_type(0).size();
                m_in_num++;
                break;
            }
            case lowered::IOExpression::io_type::OUTPUT: {
                m_io_descs[idx] = expr->get_input_port_connector(0)->get_source().get_descriptor_ptr();
                m_io_data_sizes[idx] = expr->get_node()->get_input_element_type(0).size();
                break;
            } default : {
                OPENVINO_THROW("Detected unsupported io_type");
            }
        }
        OPENVINO_ASSERT(m_io_descs[idx], "IO PortDescriptor is missed!");
        current_rank = std::max(m_io_descs[idx]->get_shape().size(), current_rank);
        idx++;
    }

    if (m_tensor_rank == 0) {
        m_tensor_rank = current_rank;
    }
}

void RuntimeConfigurator::init_data_offsets() {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::init_data_offsets")
    for (size_t i = 0; i < m_io_num; ++i) {
        offset_calculation(m_io_descs[i], m_io_data_sizes[i], i < m_in_num, m_tensor_rank, m_config.m_data_offsets[i]);
    }
}

inline void RuntimeConfigurator::offset_calculation(const lowered::PortDescriptorPtr& desc, size_t data_size, bool is_input,
                                                    size_t rank, std::vector<size_t>& offsets) {
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
    if (utils::is_dynamic_vdims(shape))
        return;

    size_t dim_step = data_size;
    offsets[offsets.size() - 1] = dim_step;

    OPENVINO_ASSERT(rank >= shape.size(), "Incorrect tensor rank!");
    const auto idx_stride = rank - shape.size();
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        dim_step *= shape[i + 1];
        offsets[i + idx_stride] = shape[i] != 1 ? dim_step : 0;
    }
    if (!layout.empty()) {
        std::vector<size_t> reordered_offsets(offsets.size());
        for (size_t i = 0; i < layout.size(); i++) {
            const auto& src_idx = is_input ? layout[i] : i;
            const auto& dst_idx = is_input ? i : layout[i];
            reordered_offsets[idx_stride + dst_idx] = offsets[idx_stride + src_idx];
        }
        offsets = std::move(reordered_offsets);
    }
}

inline void RuntimeConfigurator::LoopInitializer::init_data_ptr_shifts(const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                                       RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) {
    const auto& in_ports = loop_info->get_entry_points();
    const auto& out_ports = loop_info->get_exit_points();
    const auto in_num = in_ports.size();
    const auto out_num = out_ports.size();
    auto& ptr_increments = desc.ptr_increments;
    auto& finalization_offsets = desc.finalization_offsets;
    ptr_increments.resize(in_num + out_num);
    finalization_offsets.resize(in_num + out_num);

    RuntimeConfig::LoopDescriptorList::iterator last_execution_loop_before;
    const auto there_is_before_loop = config.get_last_executed_loop_desc_it(loop_id, m_type, last_execution_loop_before);

    auto init_shifts = [&](const std::vector<LinearIR::LoopManager::LoopPort>& loop_ports, size_t start_index) {
        for (size_t i = 0; i < loop_ports.size(); ++i) {
            const auto& loop_port = loop_ports[i];
            ptr_increments[start_index + i] = loop_port.ptr_increment;
            finalization_offsets[start_index + i] = loop_port.finalization_offset;
        }
    };
    init_shifts(in_ports, 0);
    init_shifts(out_ports, in_num);

    if (there_is_before_loop) {
        finalization_offsets = last_execution_loop_before->finalization_offsets;
        std::fill(last_execution_loop_before->finalization_offsets.begin(), last_execution_loop_before->finalization_offsets.end(), 0);
    }
}

inline void RuntimeConfigurator::LoopInitializer::init_data_sizes(const LinearIR::LoopManager::LoopInfoPtr& loop_info, RuntimeConfig::LoopDescriptor& desc) {
    const auto& in_ports = loop_info->get_entry_points();
    const auto& out_ports = loop_info->get_exit_points();
    const auto in_num = in_ports.size();
    const auto out_num = out_ports.size();
    auto& data_sizes = desc.data_sizes;
    data_sizes.resize(in_num + out_num);

    auto init_data_size = [&](const std::vector<LinearIR::LoopManager::LoopPort>& loop_ports, size_t start_index) {
        for (size_t i = 0; i < loop_ports.size(); ++i) {
            data_sizes[start_index + i] = loop_ports[i].data_size;
        }
    };
    init_data_size(in_ports, 0);
    init_data_size(out_ports, in_num);
}

bool RuntimeConfigurator::FirstLoopInitializer::is_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    return !loop_info->get_handlers().get_first_iter_handlers().empty() &&
           (utils::is_dynamic_value(loop_info->get_work_amount()) || loop_info->get_work_amount() >= loop_info->get_increment());
}

void RuntimeConfigurator::FirstLoopInitializer::init_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                                                                const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                                RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) {
    if (utils::is_dynamic_value(loop_info->get_work_amount())) {
        desc.work_amount = loop_info->get_work_amount();
    } else {
        desc.work_amount = loop_info->get_increment();
        loop_info->set_work_amount(loop_info->get_work_amount() - loop_info->get_increment());
    }
    desc.increment = loop_info->get_increment();

    init_data_ptr_shifts(loop_info, loop_id, desc, config);
    init_data_sizes(loop_info, desc);
}

void RuntimeConfigurator::FirstLoopInitializer::update_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                                                                  const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                                  RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) {
    loop_info->set_work_amount(loop_info->get_work_amount() - loop_info->get_increment());

    init_data_ptr_shifts(loop_info, loop_id, desc, config);
}

bool RuntimeConfigurator::MainLoopInitializer::is_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    // If there is First Iter loop - work_amount is already updated here
    return utils::is_dynamic_value(loop_info->get_work_amount()) || loop_info->get_work_amount() >= loop_info->get_increment();
}

void RuntimeConfigurator::MainLoopInitializer::init_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                                                               const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                               RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) {
    desc.work_amount = loop_info->get_work_amount();
    desc.increment = loop_info->get_increment();

    init_data_ptr_shifts(loop_info, loop_id, desc, config);
    init_data_sizes(loop_info, desc);
}

void RuntimeConfigurator::MainLoopInitializer::update_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                                                                 const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                                 RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) {
    desc.work_amount = loop_info->get_work_amount();

    init_data_ptr_shifts(loop_info, loop_id, desc, config);
}

bool RuntimeConfigurator::LastLoopInitializer::is_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    const auto is_dynamic = utils::is_dynamic_value(loop_info->get_work_amount());
    return (is_dynamic && loop_info->get_increment() > 1) || (!is_dynamic && loop_info->get_work_amount() % loop_info->get_increment() != 0);
}

void RuntimeConfigurator::LastLoopInitializer::init_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                                                               const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                               RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) {
    const auto is_dynamic = utils::is_dynamic_value(loop_info->get_work_amount());
    desc.work_amount = is_dynamic ? loop_info->get_work_amount() : loop_info->get_work_amount() % loop_info->get_increment();
    desc.increment = is_dynamic ? 1 : desc.work_amount;

    init_data_ptr_shifts(loop_info, loop_id, desc, config);
    init_data_sizes(loop_info, desc);
}

void RuntimeConfigurator::LastLoopInitializer::update_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                                                                 const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                                 RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) {
    desc.work_amount = loop_info->get_work_amount() % loop_info->get_increment();

    init_data_ptr_shifts(loop_info, loop_id, desc, config);
}

} // namespace lowered
} // namespace snippets
} // namespace ov
