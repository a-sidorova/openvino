// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/memory_manager.hpp"
#include "snippets/utils.hpp"
#include <snippets/itt.hpp>


namespace ngraph {
namespace snippets {

MemoryManager::MemoryManager(const std::shared_ptr<ov::Model>& model) {
    // Initialize edges and nodes
    init_edges(model);
    // Initialize boxes for MemorySolver
    init_boxes();
}

void MemoryManager::init_edges(const std::shared_ptr<ov::Model>& model) {
    int64_t order = 0;
    const auto ordered_ops = model->get_ordered_ops();
    for (const auto& op : ordered_ops) {
        if (ov::is_type<ngraph::op::v0::Constant>(op) ||
            ov::is_type<ngraph::op::v0::Parameter>(op) ||
            ov::is_type<ngraph::op::v0::Result>(op) ||
            ov::is_type<op::LoopBegin>(op))
            continue;

        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            pass::SetTopologicalOrder(buffer, order++);
            edge_clusters.push_back(BufferCluster{buffer}); // TODO: Add support of inplace
            continue;
        }
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(op)) {
            // LoopBegin should have the same order as the corresponding LoopEnd
            const auto loop_begin = loop_end->get_loop_begin();
            pass::SetTopologicalOrder(loop_begin, order);
            pass::SetTopologicalOrder(op, order++);
            continue;
        }

        bool is_node = false;  // Meaning in MemoryManager bounds
        for (size_t i = 0; i < op->get_input_size() && !is_node; ++i) {
            is_node = is_node || ov::is_type<op::Buffer>(op->get_input_node_shared_ptr(i));
        }
        for (size_t i = 0; i < op->get_output_size() && !is_node; ++i) {
            const auto target_consumers = op->get_output_target_inputs(i);
            for (const auto& in : target_consumers) {
                if (ov::is_type<op::Buffer>(in.get_node())) {
                    is_node = true;
                    break;
                }
            }
        }

        if (is_node) {
            pass::SetTopologicalOrder(op, order++);
        }
    }
}

void MemoryManager::init_boxes() {
    for (int i = 0; i < edge_clusters.size(); i++) {
        MemorySolver::Box box = { std::numeric_limits<int>::max(), 0, 0, i };
        int64_t boxSize = 0;
        for (const auto &edge : edge_clusters[i]) {
            int e_start, e_finish;
            const auto target_consumers = edge->get_output_target_inputs(0);
            if (ov::is_type<op::AllocationBuffer>(edge)) {
                e_finish = pass::GetTopologicalOrder(target_consumers.begin()->get_node()->shared_from_this());
                e_start = e_finish;  // AllocationBuffer doesn't have parent (except Constant with shape)
            } else if (ov::is_type<op::IntermediateBuffer>(edge)) {
                e_start = pass::GetTopologicalOrder(edge->get_input_node_shared_ptr(0));
                e_finish = pass::GetTopologicalOrder(target_consumers.begin()->get_node()->shared_from_this());
            }

            // TODO: Added support of Dynamic Buffers
            NGRAPH_CHECK(edge->get_output_partial_shape(0).is_static(), "MemoryManager supports only static buffers");
            auto e_size = static_cast<int64_t>(edge->get_byte_size());
            boxSize = std::max(e_size, boxSize);

            box.start = std::min(e_start, box.start);
            box.finish = std::max(e_finish, box.finish);
        }

        box.size = utils::div_up(boxSize, alignment);
        boxes.push_back(box);
    }
}

void MemoryManager::set_offset(const std::shared_ptr<op::Buffer>& buffer, const size_t offset) const {
    auto propagate_offset = [](const std::shared_ptr<ngraph::snippets::op::Buffer>& buffer, const size_t offset) {
        // If Buffer has offset We set this offset in the next Load and Store ops
        // to correctly read and write data because all buffers have the one register
        // Also if user sets offset to a Buffer It means that the Buffer has the corresponding Load and Store ops

        // Propagate to up: in Store. Buffer can have only one Store
        {
            auto parent = buffer->get_input_node_shared_ptr(0);
            if (!ov::is_type<ngraph::op::v0::Constant>(parent)) {
                auto idx = buffer->input(0).get_source_output().get_index();
                while (ov::is_type<snippets::op::LoopBase>(parent)) {
                    const auto source_output = parent->input_value(idx);
                    parent = source_output.get_node_shared_ptr();
                    idx = source_output.get_index();
                }
                if (auto memory_access = ov::as_type_ptr<ngraph::snippets::op::MemoryAccess>(parent)) {
                    auto &out_desc = memory_access->get_output_port_descriptor(idx);
                    out_desc.m_offset = offset;
                } else {
                    throw ngraph_error(
                            "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
                }
            }
        }

        // Propagate to down: in Load. Buffer can have several Load and Loops after himself. We should go through all target inputs
        {
            std::function<void(const Input<Node>&)> propagate_down;
            propagate_down = [&](const Input<Node>& target_input) {
                const auto child = target_input.get_node()->shared_from_this();
                // There may be graph with several LoopBegin and LoopEnd between Load/Brgemm and Buffer,
                // so we should iterate through LoopBase
                // Example: Softmax decomposition with ReduceMax
                if (ov::is_type<snippets::op::LoopBase>(child)) {
                    const auto index = target_input.get_index();
                    for (const auto loop_target_output : child->output(index).get_target_inputs()) {
                        propagate_down(loop_target_output);
                    }
                } else if (auto memory_access = ov::as_type_ptr<ngraph::snippets::op::MemoryAccess>(child)) {
                    auto& in_desc = memory_access->get_input_port_descriptor(target_input.get_index());
                    in_desc.m_offset = offset;
                } else {
                    throw ngraph_error("Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
                }
            };

            for (const auto target_output : buffer->output(0).get_target_inputs()) {
                propagate_down(target_output);
            }
        }
    };

    buffer->set_offset(offset);
    propagate_offset(buffer, offset);
}

int64_t MemoryManager::allocate() const {
    MemorySolver staticMemSolver(boxes);
    size_t total_size = static_cast<size_t>(staticMemSolver.solve()) * alignment;

    // Set offsets for Buffers (edges)
    for (auto& box : boxes) {
        for (auto& edge : edge_clusters[box.id]) {
            int64_t offset = staticMemSolver.getOffset(box.id);
            set_offset(edge, offset * alignment);  // alignment in byte
        }
    }

    return total_size;
}

}// namespace snippets
}// namespace ngraph
