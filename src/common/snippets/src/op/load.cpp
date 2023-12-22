// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/load.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace op {

Load::Load(const Output<Node>& x, const size_t count, const size_t offset)
    : MemoryAccess({x}, std::set<size_t>{0}, std::set<size_t>{}) {
    set_input_port_descriptor({count, offset}, 0);
    constructor_validate_and_infer_types();
}

void Load::validate_memory_access_params() const {
    // Load has memory access port only on output
    const auto input_ma_ports = get_memory_access_input_ports();
    const auto output_ma_ports = get_memory_access_output_ports();
    OPENVINO_ASSERT(input_ma_ports.size() == 1 && is_memory_access_input_port(0), "Load node must have memory access input port");
    OPENVINO_ASSERT(output_ma_ports.size() == 0, "Load node mustn't have memory access output port");
}

void Load::validate_and_infer_types() {
    validate_memory_access_params();
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> Load::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Load);
    check_new_args_count(this, new_args);
    return std::make_shared<Load>(new_args.at(0), get_count(), get_offset());
}

LoadReshape::LoadReshape(const Output<ov::Node>& x, const size_t count, const size_t offset, std::vector<size_t> order)
                            : Load(x, count, offset) {
    const auto& in_shape = x.get_partial_shape();
    OPENVINO_ASSERT(in_shape.is_static(), "LoadReshape supports only static input shapes");
    const auto in_shape_size = in_shape.size();
    OPENVINO_ASSERT(order.size() == in_shape_size, "LoadReshape got new_order of invalid size");
    OPENVINO_ASSERT(*std::max_element(order.begin(), order.end()) == in_shape_size - 1 &&
                 *std::min_element(order.begin(), order.end()) == 0, "LoadReshape detected invalid values in new_order");
    const std::set<size_t> unique_dims(order.begin(), order.end());
    OPENVINO_ASSERT(unique_dims.size() == order.size(), "LoadReshape order must not contain repeated elements");
    set_input_order(std::move(order));
    constructor_validate_and_infer_types();
}

void LoadReshape::validate_and_infer_types() {
    validate_memory_access_params();
    const auto& old_shape = get_input_partial_shape(0);
    ov::PartialShape new_shape;
    for (const auto idx : get_input_order(0))
        new_shape.push_back(old_shape[idx]);
    set_output_type(0, get_input_element_type(0), new_shape);
}

std::shared_ptr<Node> LoadReshape::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LoadReshape);
    check_new_args_count(this, new_args);
    return std::make_shared<LoadReshape>(new_args.at(0), get_count(), get_offset(), get_input_order(0));
}
LoadReshape::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& loadReshape = ov::as_type_ptr<LoadReshape>(n);
    OPENVINO_ASSERT(loadReshape, "Got invalid node in LoadReshape::ShapeInfer");
    m_order = loadReshape->get_input_order(0);
}
IShapeInferSnippets::Result LoadReshape::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got unexpected number of input shapes");
    return {{utils::get_planar_vdims(input_shapes[0], m_order)}, ShapeInferStatus::success};
}
}// namespace op
}// namespace snippets
}// namespace ov
