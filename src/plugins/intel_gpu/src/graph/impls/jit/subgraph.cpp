// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu/intel/jit/generator.hpp"

#include "primitive_inst.h"
#include "registry/implementation_map.hpp"
#include "register.hpp"
#include "subgraph.hpp"

#include "runtime/ocl/ocl_engine.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "plugin/transformations/snippets/lowered/set_single_kernel_work_amount.hpp"

#include "snippets/lowered/pass/optimize_domain.hpp"
#include "snippets/utils/utils.hpp"
#include "gpu_generator.hpp"

#include "runtime/ocl/ocl_kernel.hpp"
#include "common_utils/kernel_generator_base.hpp"

#include <vector>
namespace ov::intel_gpu::jit {

using namespace dnnl::impl::gpu::intel::jit;
using namespace ngen;

template <HW hw>
class VectorScaleKernelGenerator : public OpenCLCodeGenerator<hw> {
protected:
    NGEN_FORWARD_OPENCL(hw);

public:
    VectorScaleKernelGenerator() : OpenCLCodeGenerator<hw>()
    {
        // Define kernel interface for OpenCL.
        newArgument("src0", ExternalArgumentType::GlobalPtr);
        newArgument("src1", ExternalArgumentType::GlobalPtr);
        newArgument("dst", ExternalArgumentType::GlobalPtr);
        requireLocalID(1);
        requireLocalSize();
        requireSIMD((GRF::bytes(hw) == 64) ? 16 : 8);
        externalName("jit::subgraph");

        finalizeInterface();

        // auto surface_src0 = Surface(getArgumentSurfaceIfExists("src0"));     // Surface # for buffer.
        // auto surface_src1 = Surface(getArgumentSurfaceIfExists("src1"));     // Surface # for buffer.
        // auto surface_dst = Surface(getArgumentSurfaceIfExists("dst"));       // Surface # for buffer.

        auto src0_ptr = getArgument("src0");
        auto src1_ptr = getArgument("src1");
        auto dst_ptr = getArgument("dst");

        auto local_size = getLocalSize(0).uw();
        auto local_id = getLocalID(0);               // Vector of local IDs.
        auto group_id = r0.ud(1);                    // Thread group (a.k.a. workgroup) IDs are in r0.ud(1) (X) r0.ud(6) (Y) r0.ud(7) (Z)
 
        // Local variables.
        auto global_id = r12.ud(0);
        auto header = r13;
        auto temp = r11;

        auto reg_src0 = r14;
        auto reg_src1 = r15;

        // All instructions use W (NoMask) by default.
        setDefaultNoMask();

        // Enable automatic SWSB for Gen12.
        setDefaultAutoSWSB();

        // Prologue for ATS+.
        prologue();

        // Enable IEEE denormals.
        or_(1 | Switch, cr0[0], cr0[0], 0x4C0);

        // Calculate global ID = (group ID) * (local size) + (local ID for lane 0).
        mul(1, global_id, group_id, local_size);
        add(1, global_id, global_id, local_id[0]);

        shl(1, global_id, global_id, 2);
        {
            addc(1, header.ud(0), src0_ptr.ud(0), global_id);
            mov(1, temp.ud(0), acc0.ud(0));
            add(1, header.ud(1), src0_ptr.ud(1), temp.ud(0));
            load(1, reg_src0, D32 | V8T, A64, header);
        }
        {
            addc(1, header.ud(0), src1_ptr.ud(0), global_id);
            mov(1, temp.ud(0), acc0.ud(0));
            add(1, header.ud(1), src1_ptr.ud(1), temp.ud(0));
            load(1, reg_src1, D32 | V8T, A64, header);
        }

        // Do 32 byte (2 OWord) block read at offset (global ID) * sizeof(float).
        // shr<uint32_t>(1, header[2], global_id, 2);
        // load(8, reg_src0, block_oword(2), surface_src0, header);
        // load(8, reg_src1, block_oword(2), surface_src1, header);

        add<float>(8, reg_src0, reg_src0, reg_src1);

        // Store updated data.
        // Store updated reg_src0.
        //store(8, block_oword(2), surface_dst, header, reg_src0);

        {
            addc(1, header.ud(0), dst_ptr.ud(0), global_id);
            mov(1, temp.ud(0), acc0.ud(0));
            add(1, header.ud(1), dst_ptr.ud(1), temp.ud(0));
            store(1, D32 | V8T, A64, header, reg_src0);
        }

        // End thread. Must move r0 to one of r112-r127, then call threadend.
        mov<uint32_t>(8, r127, r0);
        threadend(r127);
    }
};

class SubgraphImpl : public primitive_impl {
    using primitive_impl::primitive_impl;

    using DataFlowPasses = std::vector<ov::snippets::pass::Manager::PositionedPassBase>;
    using ControlFlowPasses = std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered>;

    std::shared_ptr<ov::snippets::op::Subgraph> m_subgraph {nullptr};

public:
    explicit SubgraphImpl(const program_node& node, const kernel_impl_params& impl_params)
        : primitive_impl("jit::subgraph"), m_subgraph(node.as<subgraph>().get_primitive()->ov_subgraph->clone())  {
            const auto& engine = downcast<ocl::ocl_engine>(impl_params.get_program().get_engine());
            const auto& device = downcast<ocl::ocl_device>(*engine.get_device());
            HW hw = VectorScaleKernelGenerator<HW::Unknown>::detectHW(engine.get_cl_context().get(), device.get_device().get());
            cl::Kernel kernel;
            switch (hw) {
                case HW::Gen9:    kernel = VectorScaleKernelGenerator<HW::Gen9>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                case HW::Gen11:   kernel = VectorScaleKernelGenerator<HW::Gen11>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                case HW::Gen12LP: kernel = VectorScaleKernelGenerator<HW::Gen12LP>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                case HW::XeHP:    kernel = VectorScaleKernelGenerator<HW::XeHP>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                case HW::XeHPG:   kernel = VectorScaleKernelGenerator<HW::XeHPG>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                case HW::XeHPC:   kernel = VectorScaleKernelGenerator<HW::XeHPC>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                case HW::Xe2:     kernel = VectorScaleKernelGenerator<HW::Xe2>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                case HW::Xe3:     kernel = VectorScaleKernelGenerator<HW::Xe3>().getKernel(engine.get_cl_context().get(), device.get_device().get()); break;
                default:          OPENVINO_THROW("[GPU] Unsupported architecture");;
            }

            ocl_kernel = std::make_shared<ocl::ocl_kernel>(ocl::ocl_kernel_type(kernel, device.get_usm_helper()),
                                                           kernel.getInfo<CL_KERNEL_FUNCTION_NAME>());
 
            const auto total_elements_num = impl_params.get_input_layout().count();
            const auto simd = 8;

            kd.params.workGroups.global = {total_elements_num, 1, 1};
            kd.params.workGroups.local = {simd, 1, 1};

            // kd.params.scalars.push_back({cldnn::scalar_desc::Types::FLOAT32, 0});
            kd.params.arguments.push_back({cldnn::argument_desc::Types::INPUT, 0});
            kd.params.arguments.push_back({cldnn::argument_desc::Types::INPUT, 1});
            kd.params.arguments.push_back({cldnn::argument_desc::Types::OUTPUT, 0});
            // kd.params.arguments.push_back({cldnn::argument_desc::Types::SCALAR, 0});
        }
    
    [[nodiscard]] virtual cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const {
        cldnn::kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();

        auto intermediates = instance.get_intermediates_memories();
        args.intermediates = {intermediates.begin(), intermediates.end()};

        return args;
    }

    SubgraphImpl() : primitive_impl() {}

    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::jit::SubgraphImpl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<SubgraphImpl>(*this);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    
    void set_arguments(primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        auto args_data = get_arguments(instance);

        // Update scalars pointer
        args_data.scalars = &kd.params.scalars;

        for (const auto arg : kd.params.arguments) {
            GPU_DEBUG_TRACE_DETAIL << "Argument: type=" << static_cast<int>(arg.t) << " idx=" << arg.index << "\n";
        }

        stream.set_arguments(*ocl_kernel, kd.params, args_data);
    }

    void set_arguments(primitive_inst& /*instance*/, kernel_arguments_data& /*args*/) override {}
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override { return {}; }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, false, instance.is_output());
        }

        // If any user of the desc's users is CPU implementation or network's output, set desc as a output event (event
        // won't be nullptr)
        bool needs_completion_event = instance.needs_completion_event();

        auto& params = kd.params;
 
        const auto& gws = params.workGroups.global;
        const auto& lws = params.workGroups.local;

        GPU_DEBUG_TRACE_DETAIL << "Enqueue jit kernel : gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] " << "lws=["
                               << lws[0] << ", " << lws[1] << ", " << lws[2] << "]" << (needs_completion_event ? " has_completion_event=true" : "") << '\n';

        return stream.enqueue_kernel(*ocl_kernel, params, {}, events, needs_completion_event);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override { }

private:
    KernelData kd{};
    ocl::ocl_kernel::ptr ocl_kernel;

    static ngen::HW ngenHW2pluginHW(gpu_arch arch) {
        switch (arch) {
        case gpu_arch::gen9: return ngen::HW::Gen9;
        case gpu_arch::gen11: return ngen::HW::Gen11;
        case gpu_arch::xe_lp: return ngen::HW::XeLP;
        case gpu_arch::xe_hp: return ngen::HW::XeHP;
        case gpu_arch::xe_hpg: return ngen::HW::XeHPG;
        case gpu_arch::xe_hpc: return ngen::HW::XeHPC;
        case gpu_arch::xe2: return ngen::HW::Xe2;
        case gpu_arch::xe3: return ngen::HW::Xe3;
        case gpu_arch::unknown: return ngen::HW::Unknown;
        default:
            OPENVINO_THROW("Unexpected arch");
        }
    }

    static ov::snippets::op::Subgraph::BlockedShapeVector getSnippetsBlockedShapes(const kernel_impl_params& impl_params) {
        ov::snippets::op::Subgraph::BlockedShapeVector in_blocked_shapes(impl_params.input_layouts.size());
        for (size_t i = 0; i < in_blocked_shapes.size(); i++) {
            // support only planar shapes
            const auto blocked_dims = ov::snippets::utils::pshape_to_vdims(impl_params.input_layouts[i].get_partial_shape());
            const auto blocked_layout = ov::snippets::utils::get_planar_layout(blocked_dims.size());
            in_blocked_shapes[i] = {blocked_dims, blocked_layout};
        }
        return in_blocked_shapes;
    }

    static std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> getIOPrecisions(const kernel_impl_params& impl_params) {
        std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> prc;
        prc.first.reserve(impl_params.input_layouts.size());
        prc.second.reserve(impl_params.output_layouts.size());
        for (const auto& in : impl_params.input_layouts) {
            prc.first.push_back(in.data_type);
        }
        for (const auto& out : impl_params.output_layouts) {
            prc.second.push_back(out.data_type);
        }
        return prc;
    }
};

std::unique_ptr<primitive_impl> Subgraph::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<subgraph>());
    return std::make_unique<SubgraphImpl>(node, params);
}

}  // namespace ov::intel_gpu::jit

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::jit::SubgraphImpl)
