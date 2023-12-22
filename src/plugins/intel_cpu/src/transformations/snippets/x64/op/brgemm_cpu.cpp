// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "utils/general_utils.h"
#include "snippets/utils.hpp"


namespace ov {
namespace intel_cpu {

BrgemmCPU::BrgemmCPU(const Output<Node>& A, const Output<Node>& B, const Type type,
                     const size_t offset_a, const size_t offset_b, const size_t offset_c,
                     std::vector<size_t> order_a, std::vector<size_t> order_b, std::vector<size_t> order_c,
                     const size_t blk_size_m, const size_t blk_size_k, const size_t blk_size_n, const float beta)
    : Brgemm(), m_type(type), m_beta(beta) {
    // We call default ctor of Brgemm class to avoid incorrect shape infer in constructor_validate_and_type_infer() call
    set_arguments({A, B});
    set_output_size(1);
    ctor_initialize(std::set<size_t>{0, 1}, std::set<size_t>{0});
    set_input_port_descriptor(MemoryAccess::PortDescriptor(0, offset_a, std::move(order_a)), 0);
    set_input_port_descriptor(MemoryAccess::PortDescriptor(0, offset_b, std::move(order_b)), 1);
    set_output_port_descriptor(MemoryAccess::PortDescriptor(0, offset_c, std::move(order_c)), 0);
    compute_block_size_values(blk_size_m, blk_size_k, blk_size_n);
}

BrgemmCPU::BrgemmCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch, const Type type,
                     const size_t offset_a, const size_t offset_b, const size_t offset_scratch, const size_t offset_c,
                     std::vector<size_t> order_a, std::vector<size_t> order_b, std::vector<size_t> order_c,
                     const size_t blk_size_m, const size_t blk_size_k, const size_t blk_size_n, const float beta)
    : Brgemm(), m_type(type), m_beta(beta) {
    set_arguments({A, B, scratch});
    set_output_size(1);
    ctor_initialize(std::set<size_t>{0, 1, 2}, std::set<size_t>{0});
    set_input_port_descriptor(MemoryAccess::PortDescriptor(0, offset_a, std::move(order_a)), 0);
    set_input_port_descriptor(MemoryAccess::PortDescriptor(0, offset_b, std::move(order_b)), 1);
    set_input_port_descriptor(MemoryAccess::PortDescriptor(0, offset_scratch), 2);
    set_output_port_descriptor(MemoryAccess::PortDescriptor(0, offset_c, std::move(order_c)), 0);
    compute_block_size_values(blk_size_m, blk_size_k, blk_size_n);
}

BrgemmCPU::BrgemmCPU(const Output<Node>& A, const Output<Node>& B, const Type type,
                     const PortDescriptor& desc_a, const PortDescriptor& desc_b, const PortDescriptor& desc_c,
                     const size_t blk_size_m, const size_t blk_size_k, const size_t blk_size_n, const float beta)
    : Brgemm(), m_type(type), m_beta(beta) {
    set_arguments({A, B});
    set_output_size(1);
    m_input_ports = {{0, desc_a}, {1, desc_b}};
    m_output_ports = {{0, desc_c}};
    compute_block_size_values(blk_size_m, blk_size_k, blk_size_n);
}

BrgemmCPU::BrgemmCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch, const Type type,
                     const PortDescriptor& desc_a, const PortDescriptor& desc_b, const PortDescriptor& desc_scratch, const PortDescriptor& desc_c,
                     const size_t blk_size_m, const size_t blk_size_k, const size_t blk_size_n, const float beta)
    : Brgemm(), m_type(type), m_beta(beta) {
    set_arguments({A, B, scratch});
    set_output_size(1);
    m_input_ports = {{0, desc_a}, {1, desc_b}, {2, desc_scratch}};
    m_output_ports = {{0, desc_c}};
    compute_block_size_values(blk_size_m, blk_size_k, blk_size_n);
}

void BrgemmCPU::compute_block_size_values(const size_t blk_size_m, const size_t blk_size_k, const size_t blk_size_n) {
    const auto input_shape_0 = get_input_planar_partial_shape(0).get_shape();
    const auto input_shape_1 = get_input_planar_partial_shape(1).get_shape();
    m_M_blk = blk_size_m != 0 ? blk_size_m : *(input_shape_0.rbegin() + 1);
    m_K_blk = blk_size_k != 0 ? blk_size_k : *input_shape_0.rbegin();
    m_N_blk = blk_size_n != 0 ? blk_size_n : *input_shape_1.rbegin();
}

void BrgemmCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCPU_validate_and_infer_types);
    validate_inputs();

    const auto brgemm_copy = is_with_data_repacking() ? get_brgemm_copy() : nullptr;

    auto output_shape = get_output_partial_shape({ get_input_planar_partial_shape(0),
                                                   brgemm_copy ? brgemm_copy->get_input_planar_partial_shape(0)
                                                               : get_input_planar_partial_shape(1) });
    auto reordered_out_shape = output_shape;
    snippets::utils::ordered_vector(output_shape, get_output_order(0), true, reordered_out_shape);
    set_output_type(0, get_output_type(), reordered_out_shape);

    // Additional check for 3rd input
    validate_with_scratchpad();
}

void BrgemmCPU::validate_with_scratchpad() const {
    // Additional check for 3rd input
    if (one_of(m_type, Type::WithCompensations, Type::AMX)) {
        const auto& pshape = get_input_partial_shape(2);
        OPENVINO_ASSERT(pshape.is_static(), "BRGEMM Scratch must have static shape");
        if (is_with_compensations()) {
            OPENVINO_ASSERT(get_input_element_type(2) == ov::element::f32, "BRGEMM Scratch with compensations must have FP32 element type");
        }
    }
}

void BrgemmCPU::validate_inputs() const {
    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "BrgemmCPU currently supports only static shapes.");
    OPENVINO_ASSERT(implication(one_of(m_type, Type::Floating, Type::WithDataRepacking), get_input_size() == 2),
                    "BrgemmCPU expects 2 inputs in cases, when input precisions are f32|f32, u8|i8 or bf16|bf16 (non-AMX system)");
    OPENVINO_ASSERT(implication(one_of(m_type, Type::WithCompensations, Type::AMX), get_input_size() == 3),
                    "BrgemmCPU expects 3 inputs with input precisions i8|i8 and bf16|bf16 on AMX system");
}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    std::shared_ptr<BrgemmCPU> brgemm;
    if (!is_with_scratchpad()) {
        return std::make_shared<BrgemmCPU>(new_args.at(0), new_args.at(1), m_type,
                                           get_input_port_descriptor(0), get_input_port_descriptor(1), get_output_port_descriptor(0),
                                           m_M_blk, m_K_blk, m_N_blk, m_beta);
    } else {
        return std::make_shared<BrgemmCPU>(new_args.at(0), new_args.at(1), new_args.at(2), m_type,
            get_input_port_descriptor(0), get_input_port_descriptor(1), get_input_port_descriptor(2), get_output_port_descriptor(0),
            m_M_blk, m_K_blk, m_N_blk, m_beta);
    }
}

std::shared_ptr<BrgemmCopyB> BrgemmCPU::get_brgemm_copy() const {
    OPENVINO_ASSERT(one_of(m_type, Type::WithDataRepacking, Type::WithCompensations, Type::AMX), "Brgemm doesn't need BrgemmCopyB");
    auto b_input_node = get_input_node_shared_ptr(1);
    if (const auto brgemm_copy_b = ov::as_type_ptr<BrgemmCopyB>(b_input_node)) {
        return brgemm_copy_b;
    }
    if (ov::is_type<snippets::op::IntermediateMemoryBuffer>(b_input_node)) {
        if (const auto brgemm_copy_b = ov::as_type_ptr<BrgemmCopyB>(b_input_node->get_input_node_shared_ptr(0))) {
            return brgemm_copy_b;
        }
    }
    OPENVINO_THROW("BrgemmCopyB hasn't been found!");
}

size_t BrgemmCPU::get_offset_scratch() const {
    OPENVINO_ASSERT(is_with_scratchpad() && get_input_size() == 3, "Offset of scratchpad must be only in Brgemm with scratchpad on 3rd input");
    return get_input_offset(2);
}

bool BrgemmCPU::visit_attributes(AttributeVisitor& visitor) {
    Brgemm::visit_attributes(visitor);
    visitor.on_attribute("blk_M", m_M_blk);
    visitor.on_attribute("blk_K", m_K_blk);
    visitor.on_attribute("blk_N", m_N_blk);
    visitor.on_attribute("beta", m_beta);
    return true;
}

} // namespace intel_cpu
} // namespace ov
