// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "jit_snippets_emitters.hpp"
#include "snippets/op/matmul_cpu.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"

using namespace Xbyak;
using ngraph::snippets::op::Subgraph;

namespace ov {
namespace intel_cpu {

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<Reg64>& regs) {
    regs.resize(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});
}

jit_container_emitter::jit_container_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                      const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_container_emitter::map_abstract_registers(mapping_info& gpr_map_pool,  mapping_info& vec_map_pool,
                            std::vector<AllocatedEmitter>& allocated_emitters) const {
    if (allocated_emitters.empty())
        IE_THROW() << "Cannot map registers when there is no allocated_emitters provided";
    auto map_regs = [](const std::vector<size_t>& abstract_regs, mapping_info& mapping) {
        auto& abstract_to_physical = mapping.first;
        auto& regs_pool = mapping.second;
        std::vector<size_t> physical_regs(abstract_regs.size());
        for (size_t i = 0; i < abstract_regs.size(); i++) {
            const auto abstract = abstract_regs[i];
            auto& physical = physical_regs[i];
            if (abstract_to_physical.count(abstract) == 0) {
                if (regs_pool.empty())
                    IE_THROW() << "Cannot map registers for jit_container_emitter: not enough regs in the pool";
                physical = regs_pool.back();
                regs_pool.pop_back();
                abstract_to_physical[abstract] = physical;
            } else {
                physical = abstract_to_physical[abstract];
            }
        }
        return physical_regs;
    };

    for (auto& code : allocated_emitters) {
        const auto& emitter = code.first;
        std::vector<size_t> in_abstract_regs, out_abstract_regs;
        std::tie(in_abstract_regs, out_abstract_regs) = code.second;
        std::vector<size_t> in_physical_regs, out_physical_regs;
        switch (std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type()) {
            case gpr_to_gpr:
                // Note that gpr_to_gpr is used for high-level utility operations like Kernel/Loop.
                // Input registers are not mapped in this case, since they contain utility info
                // (num_params, loop increment, etc.), but not reg indexes.
                // todo: Note that LoopBeginEmitter and LoopEndEmitter demonstrate new paradigm,
                //  where all utility emitters align with conventional Op emitters
                if (std::dynamic_pointer_cast<LoopBeginEmitter>(emitter) ||
                    std::dynamic_pointer_cast<LoopEndEmitter>(emitter) ||
                    std::dynamic_pointer_cast<MatMulEmitter>(emitter))
                    in_physical_regs = std::move(map_regs(in_abstract_regs, gpr_map_pool));
                else
                    in_physical_regs = std::move(in_abstract_regs);
                out_physical_regs = std::move(map_regs(out_abstract_regs, gpr_map_pool));
                break;
            case gpr_to_vec:
                // Load Emitters
                in_physical_regs = std::move(map_regs(in_abstract_regs, gpr_map_pool));
                out_physical_regs = std::move(map_regs(out_abstract_regs, vec_map_pool));
                break;
            case vec_to_gpr:
                // Store Emitters
                in_physical_regs = std::move(map_regs(in_abstract_regs, vec_map_pool));
                out_physical_regs = std::move(map_regs(out_abstract_regs, gpr_map_pool));
                break;
            case vec_to_vec:
                // Regular operations
                in_physical_regs = std::move(map_regs(in_abstract_regs, vec_map_pool));
                out_physical_regs = std::move(map_regs(out_abstract_regs, vec_map_pool));
                break;
            default:
                IE_THROW() << "Unhandled in_out type";
        }
        code.second = std::make_pair(in_physical_regs, out_physical_regs);
        if (auto container = std::dynamic_pointer_cast<jit_container_emitter>(code.first))
            container->map_abstract_registers(gpr_map_pool,  vec_map_pool, allocated_emitters);
    }
}

KernelEmitter::KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_container_emitter(h, isa, n) {
    const auto kernel = ov::as_type_ptr<ngraph::snippets::op::Kernel>(n);
    if (!kernel)
        IE_THROW() << "KernelEmitter invoked with invalid op argument";
    if (kernel->region.empty())
        IE_THROW() << "KernelEmitter invoked with empty body";
    if (kernel->compile_params == nullptr)
        IE_THROW() << "KernelEmitter invoked with op::Kernel that contains no compile_params";
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    // calc data access pattern. we'll need it for offsets calculation
    const auto&  model = kernel->model;
    const auto get_static_shape = [](const std::shared_ptr<ov::Node>& node) {
        const auto& pshape = node->get_output_partial_shape(0);
        if (pshape.is_dynamic())
            IE_THROW() << "KernelEmitter can't calc offsets for dynamic shapes";
        return pshape.get_shape();
    };
    const auto get_data_layout = [](const Output<ov::Node>& out, std::vector<size_t>& shape) {
        const auto& layout = ngraph::snippets::utils::get_port_layout(out);
        // default access pattern
        if (!layout.empty()) {
            const auto layout_shape_diff = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(layout.size());
            // Plugin can (and usually does) prepend shapes with 1's to facilitate scheduling, here we can safely remove leading 1's
            if (layout_shape_diff > 0) {
                if (std::any_of(shape.begin(), shape.begin() + layout_shape_diff, [](size_t x){return x != 1;}))
                    IE_THROW() << "KernelEmitter detected shape vs access pattern conflict: only leading 1's can be removed from the shape";
                shape.erase(shape.begin(), shape.begin() + layout_shape_diff);
            }
        }
        return layout;
    };
    auto params = model->get_parameters();
    auto results = model->get_results();
    num_inputs = params.size();
    num_outputs = results.size();
    NodeVector io_nodes;
    std::copy(params.begin(), params.end(), std::back_inserter(io_nodes));
    std::copy(results.begin(), results.end(), std::back_inserter(io_nodes));

    const auto& model_rt_info = model->get_rt_info();
    const auto& plugin_shapes = model_rt_info.find("PluginShapesOverride");
    if (plugin_shapes == model_rt_info.end()) {
        IE_THROW() << "JIT KernelEmitter requires plugin-overriden shapes in model rt_info";
    } else {
        const auto& new_shapes = plugin_shapes->second.as<std::vector<std::vector<size_t>>>();
        if (new_shapes.size() != num_inputs + num_outputs)
            IE_THROW() << "JIT KernelEmitter detected invalid plugin-overriden shapes";
        io_shapes = new_shapes;
    }
    for (int i = 0; i < io_nodes.size(); i++) {
        const auto& out = io_nodes[i]->output(0);
        data_layout.push_back(get_data_layout(out, io_shapes[i]));
        io_data_size.push_back(out.get_element_type().size());
    }
    // Initialize pools of gp and vec registers
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 16; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 15 - i;
    // todo: it's more convenient to use std::set as a pool container (unique and always sorted),
    //  but pools are vectors to align with emit_code signature. Change signature?
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                       [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, {Xbyak::Operand::RSP, Xbyak::Operand::RBP,
                                         reg_indexes_idx, reg_const_params_idx});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);
    std::vector<AllocatedEmitter> data_io_emitters;
    std::copy_if(body.begin(), body.end(), std::back_inserter(data_io_emitters),
                           [](const AllocatedEmitter& code){
                                   const auto& emitter = code.first;
                                   const auto emitter_type = std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type();
                                   // todo: how this will be handled if Matmul in & out are op::Buffer
                                   // Matmul is a special case since it incorporates input and output (we use onednn kernel)
                                   // Just like Load & Store it requires offsets calculation
                                   const auto is_matmul = std::dynamic_pointer_cast<MatMulEmitter>(emitter) != nullptr;
                                   return emitter_type == gpr_to_vec || emitter_type == vec_to_gpr || is_matmul;
                           });
    // Note that we can't use reg_indexes_idx or reg_const_params_idx to store data pointers because these two
    // regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, data_io_emitters);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);
    // However we can use reg_indexes_idx and reg_const_params_idx for other operations since we won't need them
    // after offsets calculation
    gpr_map_pool.second.push_back(reg_indexes_idx);
    gpr_map_pool.second.push_back(reg_const_params_idx);
    map_abstract_registers(gpr_map_pool, vec_map_pool, body);
}

void KernelEmitter::emit_code(const std::vector<size_t> &in,
                              const std::vector<size_t> &out,
                              const std::vector<size_t> &pool,
                              const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void KernelEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool,
                                       const std::vector<size_t> &gpr) const {
    if (!in.empty())
        IE_THROW() << "KernelEmitter got invalid number of inputs. Expected 0, got " << in.size();
    if (!out.empty())
        IE_THROW() << "KernelEmitter got invalid number of outputs. Expected 0, got " << out.size();
    const auto num_params = num_inputs + num_outputs;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    if (data_ptr_regs_idx.size() != num_params)
        IE_THROW() << "KernelEmitter arguments are inconsistent with the gpr_regs_used size: in[0] + in[1] = "
        << num_params << " data_ptr_regs_idx.size() = " << data_ptr_regs_idx.size();
}

void KernelEmitter::init_data_pointers(size_t num_inputs, size_t num_params,
                                              const Reg64& reg_indexes, const Reg64& reg_const_params, const std::vector<Reg64>& data_ptr_regs) const {
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = jcp.master_shape.size() - 1;
    //const size_t tile_rank = jcp.tile_rank;
    std::vector<std::vector<size_t>> data_offsets(num_params, std::vector<size_t>{});
    auto offset_calculation = [=](const std::vector<size_t>& shape,
                                            const std::vector<size_t>& layout, const size_t data_size) {
        // Strides represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    strides: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    strides: s1*s3, s3,       0,  1
        std::vector<size_t> strides(shape.size());
        size_t dim_step = 1;
        strides[shape.size() - 1] = 1;
        for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
            dim_step *= shape[k+1];
            strides[k] = shape[k] != 1 ? dim_step * data_size : 0;
        }
        // Note: this is an extra copy, but let's keep it for clarity
        if (!layout.empty()) {
            std::vector<size_t> reordered_strides(strides.size());
            for (auto i = 0; i < layout.size(); i++)
                reordered_strides[i] = strides[layout[i]];
            strides = std::move(reordered_strides);
        }
        // the last stride is ignored, since the entire last dim is processed by kernel
        // and no parallel_for data_ptr offsets can be applied in this case (cover tile_rank == 1)
        strides.pop_back();
        // if tile_rank > 1, then zero corresponding strides since no external offset can be applied
        // for (auto j = 0; j < tile_rank - 1; j++)
        //    strides[strides.size() - 1 - j] = 0;
        // actual offset size might be larger that the shape size due to 6D scheduling
        strides.insert(strides.begin(), offset_rank - strides.size(), 0);

        return strides;
    };
    for (size_t i = 0; i < num_params; i++) {
        data_offsets[i] = offset_calculation(io_shapes[i],  data_layout[i], io_data_size[i]);
    }
    // master_shape size must be valid in both static and dynamic cases
    std::function<void(Reg64, const std::vector<size_t>&, Reg64)> init_ptr_with_offset;
    init_ptr_with_offset = [&](Reg64 pointer, const std::vector<size_t>& offsets, Reg64 reg_tmp) {
        for (int j = 0; j < offset_rank; j++) {
            if (jcp.master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    const auto spare_corruptable_gpr = std::find_if(gp_regs_pool.begin(), gp_regs_pool.end(),
                                                   [this](size_t reg) {
                                                        return reg != reg_indexes_idx && reg != reg_const_params_idx;
                                                   });
    const bool last_iter_explicitly = spare_corruptable_gpr == gp_regs_pool.end();
    Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs.back() : Reg64(static_cast<int>(*spare_corruptable_gpr));
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_const_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_const_params to pass runtime args to LoopScheduler, so we have to
    //     push a reg on the stack, and restore it value afterwards
    if (last_iter_explicitly) {
        h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_const_params;
        // can corrupt reg_const_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
}
void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& vec_pool,
                              const std::vector<size_t>& gpr_pool,
                              const ov::intel_cpu::emitter_context *emit_context) const {
    h->preamble();

    Reg64 reg_indexes = Reg64(static_cast<int>(reg_indexes_idx));
    Reg64 reg_const_params = Reg64(static_cast<int>(reg_const_params_idx));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_regs_idx, data_ptr_regs);

    init_data_pointers(num_inputs, num_inputs + num_outputs, reg_indexes, reg_const_params, data_ptr_regs);
    for (const auto& c : body) {
        const auto& emitter = c.first;
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = c.second;
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }
    h->postamble();
}


LoopBeginEmitter::LoopBeginEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    loop_begin = ov::as_type_ptr<ngraph::snippets::op::LoopBegin>(n);
    if (!loop_begin)
        IE_THROW() << "LoopBeginEmitter invoked with invalid op argument";
    const auto& target_inputs = loop_begin->output(loop_begin->get_output_size() - 1).get_target_inputs();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (target_inputs.size() != 1)
        IE_THROW() << "LoopBeginEmitter invoked with invalid configuration: the last output must have exactly one input attached";
    const auto loop_end = ov::as_type_ptr<ngraph::snippets::op::LoopEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        IE_THROW() << "LoopBeginEmitter invoked with invalid configuration: the last output must be LoopEnd";
    work_amount = loop_begin->get_work_amount();
    evaluate_once = loop_begin->get_evaluate_once();
    num_inputs = loop_begin->get_input_size();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopBeginEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out,
                                 const std::vector<size_t> &pool,
                                 const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}

void LoopBeginEmitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out,
                                        const std::vector<size_t> &pool,
                                        const std::vector<size_t> &gpr) const {
    if (in.size() != num_inputs)
        IE_THROW() << "Invalid inputs size: expected " << num_inputs << " got " << in.size();
    if (out.size() != num_inputs + 1)
        IE_THROW() << "Invalid outputs size: expected " << num_inputs + 1 << " got " << out.size();
}

void LoopBeginEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out,
                                 const std::vector<size_t>& pool,
                                 const std::vector<size_t>& gpr,
                                 const ov::intel_cpu::emitter_context *emit_context) const {
    // todo: In dynamic case we will also need to set broadcasting info here
    Reg64 reg_work_amount = Reg64(out.back());
    Label for_body;
    // save previous register state (if there is an outer loop that uses this reg for example)
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    loop_begin->begin_address = h->getCurr();
    loop_begin->input_regs = in;
}

LoopEndEmitter::LoopEndEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    loop_end = ov::as_type_ptr<ngraph::snippets::op::LoopEnd>(n);
    if (!loop_end)
        IE_THROW() << "LoopEndEmitter invoked with invalid op argument";
    loop_begin = loop_end->get_loop_begin();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (!loop_begin)
        IE_THROW() << "LoopEndEmitter invoked with invalid configuration: the last arg must be LoopBegin";
    // Note that 1 edge connects LoopBegin and LoopEnd
    num_inputs = loop_begin->get_input_size();
    num_outputs = loop_end->get_output_size();
    wa_increment = loop_end->get_increment();
    work_amount = loop_end->get_work_amount();
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    evaluate_once = loop_end->get_evaluate_once();
    for (int i = 0; i < num_inputs; i++)
        io_data_size.push_back(static_cast<int64_t>(loop_begin->get_input_element_type(i).size()));
    for (int i = 0; i < num_outputs; i++)
        io_data_size.push_back(static_cast<int64_t>(loop_end->get_input_element_type(i).size()));
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopEndEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out,
                                 const std::vector<size_t> &pool,
                                 const std::vector<size_t> &gpr) const {
    validate_arguments(in, out, pool, gpr);
    emit_impl(in, out, pool, gpr, nullptr);
}


void LoopEndEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool,
                                       const std::vector<size_t> &gpr) const {
    if (loop_begin->input_regs.size() != num_inputs)
        IE_THROW() << "Invalid loop_begin->input_regs size: expected " << num_inputs << " got " << loop_begin->input_regs.size();
    if (out.size() != num_outputs)
        IE_THROW() << "Invalid number of out arguments: expected " << num_outputs << " got " << out.size();
    if (in.size() != num_outputs + 1)
        IE_THROW() << "Invalid number of in arguments: expected " << num_inputs + 1 << " got " << in.size();
    const auto io_size = num_inputs + num_outputs;
    if (ptr_increments.size() != io_size)
        IE_THROW() << "Invalid apply_increments size: expected " << io_size << " got " << ptr_increments.size();
    if (finalization_offsets.size() != io_size)
        IE_THROW() << "Invalid finalization_offsets size: expected: " << io_size << " got " << finalization_offsets.size();
}

void LoopEndEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out,
                                 const std::vector<size_t>& pool,
                                 const std::vector<size_t>& gpr,
                                 const ov::intel_cpu::emitter_context *emit_context) const {
    std::vector<size_t> data_ptr_reg_idxs(loop_begin->input_regs);
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(out.begin(), out.end(), std::back_inserter(data_ptr_reg_idxs));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    Reg64 reg_work_amount = Reg64(in.back());
    if (!evaluate_once) {
        for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
            if (ptr_increments[idx] != 0)
                h->add(data_ptr_regs[idx], ptr_increments[idx] * io_data_size[idx]);
        }
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(loop_begin->begin_address);
    }

    for (int idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
}

BroadcastMoveEmitter::BroadcastMoveEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    if (n->get_input_element_type(0) != n->get_output_element_type(0))
        IE_THROW() << "BroadcastMoveEmitter supports only equal input and output types but gets: "
            << n->get_input_element_type(0) << " and " << n->get_output_element_type(0);
    byte_size = n->get_input_element_type(0).size();
}

void BroadcastMoveEmitter::emit_impl(const std::vector<size_t>& in,
          const std::vector<size_t>& out,
          const std::vector<size_t>& pool,
          const std::vector<size_t>& gpr,
          const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "BroadcastMove emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void BroadcastMoveEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Xmm xmm_src0 = Xmm(in[0]);
    Vmm vmm_dst  = Vmm(out[0]);

    switch (byte_size) {
        case 4: h->uni_vbroadcastss(vmm_dst, xmm_src0); break;
        case 2: h->vpbroadcastw(vmm_dst, xmm_src0); break;
        case 1: h->vpbroadcastb(vmm_dst, xmm_src0); break;
        default: assert(!"unsupported data type");
    }
}

ScalarEmitter::ScalarEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    value = dnnl::impl::cpu::x64::float2int(ov::as_type_ptr<ngraph::snippets::op::Scalar>(n)->cast_vector<float>()[0]);
    push_arg_entry_of("scalar", value, true);
    prepare_table();
}

void ScalarEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& pool,
                              const std::vector<size_t>& gpr,
                              const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Scalar emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ScalarEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_dst  = Vmm(out[0]);
    h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
}


MemoryEmitter::MemoryEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                             const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    src_prc = InferenceEngine::details::convertPrecision(n->get_input_element_type(0));
    dst_prc = InferenceEngine::details::convertPrecision(n->get_output_element_type(0));
}

StoreEmitter::StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
        IE_THROW() << "StoreEmitter supports only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    count = ov::as_type_ptr<ngraph::snippets::op::Store>(n)->get_count();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out,
                             const std::vector<size_t>& pool,
                             const std::vector<size_t>& gpr,
                             const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Store emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    if (!store_emitter)
        IE_THROW() << "Store CPU emitter isn't initialized for StoreEmitter!";
    store_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreEmitter::emit_data() const {
    store_emitter->emit_data();
}

LoadEmitter::LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
        IE_THROW() << "LoadEmitter supports only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    count = std::dynamic_pointer_cast<ngraph::snippets::op::Load>(n)->get_count();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "Load emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    if (!load_emitter)
        IE_THROW() << "Load CPU emitter isn't initialized for LoadEmitter!";
    load_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadEmitter::emit_data() const {
    load_emitter->emit_data();
}

BroadcastLoadEmitter::BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    if (src_prc != dst_prc)
            IE_THROW() << "BroadcastEmitters support only equal input and output types but gets: " << src_prc.name() << " and " << dst_prc.name();

    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void BroadcastLoadEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out,
                                     const std::vector<size_t>& pool,
                                     const std::vector<size_t>& gpr,
                                     const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "BroadcastLoad emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void BroadcastLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(in[0]);
    Vmm vmm_dst = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    switch (src_prc.size()) {
        case 4: h->uni_vbroadcastss(vmm_dst, h->ptr[in_reg]); break;
        case 2: h->vpbroadcastw(vmm_dst, h->ptr[in_reg]); break;
        case 1: h->vpbroadcastb(vmm_dst, h->ptr[in_reg]); break;
        default: assert(!"unsupported data type");
    }
}

LoadConvertEmitter::LoadConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : MemoryEmitter(h, isa, n) {
    count = ov::as_type_ptr<ngraph::snippets::op::Load>(n)->get_count();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out,
                                   const std::vector<size_t>& pool,
                                   const std::vector<size_t>& gpr,
                                   const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "LoadConvert emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void LoadConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        IE_THROW() << "Load CPU emitter isn't initialized for LoadEmitter!";
    load_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadConvertEmitter::emit_data() const {
    load_emitter->emit_data();
}

StoreConvertEmitter::StoreConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                         const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    count = ov::as_type_ptr<ngraph::snippets::op::Store>(n)->get_count();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;

    if (ov::is_type<ov::intel_cpu::StoreConvertTruncation>(n)) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::truncation));
    } else if (ov::is_type<ov::intel_cpu::StoreConvertSaturation>(n)) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::saturation));
    }
}

void StoreConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out,
                                    const std::vector<size_t>& pool,
                                    const std::vector<size_t>& gpr,
                                    const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << "StoreConvert emitter doesn't support " << host_isa_;
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void StoreConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        IE_THROW() << "Store CPU emitter isn't initialized for StoreEmitter!";
    store_emitter->emit_code({in[0]}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreConvertEmitter::emit_data() const {
    store_emitter->emit_data();
}
size_t MatMulEmitter::getBrgIdx(size_t mIdx, size_t kIdx, size_t nIdx) const {
    return mIdx * 4 + kIdx * 2 + nIdx;
}
MatMulEmitter::MatMulEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                         const std::shared_ptr<ov::Node>& node) : jit_emitter(h, isa, node) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& matmul_node = as_type_ptr<ngraph::snippets::op::MatMulCPU>(node);
    if (matmul_node->is_dynamic())
        IE_THROW() << "Snippets don't support code generation for dynamic MatmulCPU";
    const OutputVector io_values {matmul_node->input_value(0), matmul_node->input_value(1), matmul_node->output(0)};
    std::vector<size_t> leading_dimensions;
    std::vector<std::vector<size_t>> io_layouts;
    for (const auto& val : io_values) {
        const auto& layout = ngraph::snippets::utils::get_port_layout(val);
        const auto& io_shape = val.get_shape();
        if (layout.empty()) {
            // empty value indicates a planar layout
            leading_dimensions.push_back(io_shape.back());
            std::vector<size_t> default_layout(io_shape.size());
            std::iota(default_layout.begin(), default_layout.end(), 0);
            io_layouts.push_back(default_layout);
        } else {
            // The idea here is to find "2" (for 4D shapes) in the layout and multiply dimensions that are to the right
            // This implies that "3" is the last layout value, otherwise this layout is not supported.
            // counting from the end since shape could be prepended with ones
            const int64_t num_last_dims = layout.end() - std::find(layout.begin(), layout.end(), layout.size() - 2) - 1;
            if (layout.back() != layout.size() - 1 || num_last_dims < 1)
                IE_THROW() << "MatMulEmitter detected invalid layout values: " <<
                    "check that this shape + layout combination is schedulable";
            leading_dimensions.emplace_back(
                    std::accumulate(io_shape.end() - num_last_dims, io_shape.end(), 1, std::multiplies<size_t>()));
            io_layouts.push_back(layout);
        }
    }
    // todo: leave AMX and VNNI related code for now, it'll help to enable int8 and bf16 support
    bool isAMXSupported = mayiuse(avx512_core_bf16_amx_int8) || mayiuse(avx512_core_bf16_amx_bf16);

    const auto& A_shape = io_values[0].get_shape();
    const auto& A_layout = io_layouts[0];
    const auto& C_shape = io_values[2].get_shape();
    const auto& C_layout = io_layouts[2];
    // Batch could be broadcasted, so must be read from the out shape
    batch0 = C_shape[C_layout[0]];
    batch1 = C_shape[C_layout[1]];

    M = C_shape[C_layout[2]];
    K0 = A_shape[A_layout[3]];
    M_blk = matmulOptimalM;
    M_tail = M % M_blk;
    // B_shape[B_layout[3]]
    N0 = C_shape[C_layout[3]];

    auto brg0Prc = InferenceEngine::details::convertPrecision(matmul_node->get_input_element_type(0));
    auto brg1Prc = InferenceEngine::details::convertPrecision(matmul_node->get_input_element_type(1));
    io_data_size = {brg0Prc.size(), brg1Prc.size(), matmul_node->get_output_element_type(0).size()};
    brg0VnniFactor = 4 / brg0Prc.size();
    bool brg0WithAMX = isAMXSupported && brg0Prc != Precision::FP32 && (K0 % brg0VnniFactor == 0) && (N0 % brg0VnniFactor == 0);

    N0_blk = brg0Prc == Precision::FP32 ? N0 :
             brg0Prc == Precision::BF16 ? 32 : 64;
    N0_tail = N0 % N0_blk;
    K0_blk = brg0WithAMX ? brg0Prc == Precision::BF16 ? 32 : 64
                         : K0;
    K0_tail = K0 % K0_blk;

    size_t brg0BaseIdx = -1;
    for (size_t m = 0; m < 2; m++) {
        for (size_t k = 0; k < 2; k++) {
            for (size_t n = 0; n < 2; n++) {
                auto& brgemmCtx = brgCtxs0[getBrgIdx(m, k, n)];

                auto M_ = m ? M_tail
                            : M < M_blk ? 0 : M_blk;
                auto N_ = n ? N0_tail : N0 - N0_tail;
                auto K_ = k ? K0_tail : K0 - K0_tail;
                auto beta = k && brgCtxs0[getBrgIdx(m, 0, n)].K != 0 ? 1.0f : 0.0f;

                brgemmCtx.M = M_;
                brgemmCtx.N = N_;
                brgemmCtx.K = K_;
                brgemmCtx.LDA = leading_dimensions[0];
                brgemmCtx.LDB = leading_dimensions[1];
                brgemmCtx.LDC = leading_dimensions[2];
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg0Prc));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg1Prc));
                brgemmCtx.beta = beta;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg0BaseIdx == -1)
                        brg0BaseIdx = getBrgIdx(m, k, n);
                    initBrgemm(brgemmCtx, brgKernels0[getBrgIdx(m, k, n)], brg0WithAMX);
                }
            }
        }
    }
}

void MatMulEmitter::initBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, bool use_amx) const {
    brgemm_t brgDesc;
    brgemm_strides_t strides {static_cast<dnnl_dim_t>(ctx.M * ctx.K), static_cast<dnnl_dim_t>(ctx.K * ctx.N)};

    const bool is_int8 = utils::one_of(ctx.dt_in0, data_type::u8, data_type::s8) && utils::one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    auto isa = use_amx ? isa_any
                       : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : (is_int8 ? avx512_core_vnni : avx512_core);
    auto status = brgemm_desc_init(&brgDesc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
                                   false, false, brgemm_row_major, 1.f, ctx.beta, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, &strides);
    if (status != dnnl_success)
        IE_THROW() << "MatMulEmitter cannot initialize brgemm descriptor due to invalid params";

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(brgDesc, ctx.palette);
    if (use_amx)
        amx_tile_configure(ctx.palette);

    ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success)
        IE_THROW() << "MatMulEmitter cannot initialize brgemm kernel due to invalid params";
    brgKernel.reset(brgKernel_);
}

void MatMulEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out,
                              const std::vector<size_t>& pool,
                              const std::vector<size_t>& gpr,
                              const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == cpu::x64::sse41) {
        emit_isa<cpu::x64::sse41>(in, out);
    } else if (host_isa_ == cpu::x64::avx2) {
        emit_isa<cpu::x64::avx2>(in, out);
    } else if (host_isa_ == cpu::x64::avx512_core) {
        emit_isa<cpu::x64::avx512_core>(in, out);
    } else {
        assert(!"unsupported isa");
    }
}
template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void MatMulEmitter::emit_brgemm_kernel_call(const brgemm_kernel_t *brgKernel, int bs,
                                   Reg64 addr_A, Reg64 addr_B,
                                   const brgemm_batch_element_t *batch, Reg64 addr_C, void *scratch) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    size_t gpr_size = 8;
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                     h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    // caller obligation to save k-regs as callee may use them
    size_t n_k_regs_to_save = 8;
    if (isa == cpu::x64::avx512_core) {
        h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
        for (size_t i = 0; i < n_k_regs_to_save; ++i) {
            if (mayiuse(avx512_core))
                h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
            else
                h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
        }
    }

    // 1. Caller obligation to save vector registers as callee may use them.
    // 2. There is an implicit assumption that the host code uses the same
    // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
    // `vlen` should be replaced with `host_isa::vlen` and
    // `host_isa::vecs_count`.
    h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
    for (size_t i = 0; i < get_max_vecs_count(); ++i)
        h->uni_vmovups(h->ptr[h->rsp + i * get_vec_length()], Vmm(i));

    // save function address in gpr to pass in call instruction
    const auto& brgemm_kernel_overload =   static_cast<void (*)(const brgemm_kernel_t*,
                                                                int,
                                                                const void*,
                                                                const void*,
                                                                const brgemm_batch_element_t*,
                                                                void*,
                                                                void*)>(brgemm_kernel_execute);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(brgemm_kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), addr_A);
    h->uni_vmovq(Xmm(1), addr_B);
    h->uni_vmovq(Xmm(2), addr_C);
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(brgKernel));
    h->mov(abi_param2, bs);
    h->uni_vmovq(abi_param3, Xmm(0));
    h->uni_vmovq(abi_param4, Xmm(1));
    size_t num_args_passed_on_stack = 1;
#ifdef _WIN32
    num_args_passed_on_stack = 3;
    h->sub(h->rsp, gpr_size * num_args_passed_on_stack);
    h->sub(h->rsp, gpr_size);
    h->mov(h->qword[h->rsp], reinterpret_cast<uint64_t>(scratch));
    h->mov(h->qword[h->rsp + gpr_size], reinterpret_cast<uintptr_t>(batch));
    h->mov(h->qword[h->rsp + 2 * gpr_size], Xmm(2));
#else
    h->mov(abi_param5, reinterpret_cast<uintptr_t>(batch));
    h->uni_vmovq(abi_param6, Xmm(2));
    h->sub(h->rsp, gpr_size);
    h->mov(h->qword[h->rsp], reinterpret_cast<uint64_t>(scratch));
#endif
   // align stack on 16-byte as ABI requires
   // note that RBX must not be changed by the callee
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);

    h->call(h->rbp);

    h->add(h->rsp, h->rbx);
    h->add(h->rsp, gpr_size * num_args_passed_on_stack);
    // restore vector registers
    for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
        h->uni_vmovups(Vmm(i), h->ptr[h->rsp + i * get_vec_length()]);
    }
    h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

    // restore k registers
    if (isa == cpu::x64::avx512_core) {
        for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
            if (mayiuse(avx512_core))
                h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
            else
                h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        }
        h->add(h->rsp, n_k_regs_to_save * k_mask_size);
    }

    // restore gpr registers
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void MatMulEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu::x64::sse41, Xmm, isa == cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 input_0(static_cast<int>(in[0]));
    Reg64 input_1(static_cast<int>(in[1]));
    Reg64 output_0(static_cast<int>(out[0]));

    for (size_t mb = 0; mb < div_up(M, M_blk); mb++) {
        const bool is_M_tail = (M - mb * M_blk < M_blk);

        size_t brgIdx0 = getBrgIdx(0, 0, 0);
        size_t K0_step0 = brgCtxs0[brgIdx0].K;
        size_t K0_step1 = brgCtxs0[brgIdx0].K * brgCtxs0[brgIdx0].LDB;
        size_t N0_step0 = brgCtxs0[brgIdx0].N * brg0VnniFactor;
        size_t N0_step1 = brgCtxs0[brgIdx0].N;
        for (size_t n = 0; n < 2; n++) {
            for (size_t k = 0; k < 2; k++) {
                size_t mIdx = is_M_tail ? 1 : 0;
                auto& brgemmCtx = brgCtxs0[getBrgIdx(mIdx, k, n)];

                if (brgemmCtx.K != 0 && brgemmCtx.N != 0) {
                    const size_t in0_offset = (k * K0_step0 + mb * M_blk * brgemmCtx.LDA) * io_data_size[0];
                    const size_t in1_offset = (k * K0_step1 + n * N0_step0) * io_data_size[1];
                    const size_t out0_offset = (n * N0_step1 + mb * M_blk * brgemmCtx.LDC) * io_data_size[2];
                    if (in0_offset != 0)
                        h->add(input_0, in0_offset);
                    if (in1_offset != 0)
                        h->add(input_1, in1_offset);
                    if (out0_offset != 0)
                        h->add(output_0, out0_offset);
                    emit_brgemm_kernel_call<isa>(brgKernels0[getBrgIdx(mIdx, k, n)].get(),
                                                 1,
                                                 input_0,
                                                 input_1,
                                                 nullptr,
                                                 output_0,
                                                 nullptr);
                    if (in0_offset != 0)
                        h->sub(input_0, in0_offset);
                    if (in1_offset != 0)
                        h->sub(input_1, in1_offset);
                    if (out0_offset != 0)
                        h->sub(output_0, out0_offset);
                }
            }
        }
    }
}
}   // namespace intel_cpu
}   // namespace ov
