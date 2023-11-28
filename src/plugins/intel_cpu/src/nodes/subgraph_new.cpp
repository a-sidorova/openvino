// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "subgraph_new.h"

#include "snippets/op/subgraph.hpp"
#include "snippets/pass/hash.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/lowered/pass/pass.hpp"

#include "emitters/x64/snippets/cpu_generator.hpp"

#include "transformations/defs.hpp"
#include "transformations/snippets/x64/pass/lowered/set_brgemm_copy_b_buffers_shape.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_blocking.hpp"
#include "transformations/snippets/x64/pass/mul_add_to_fma.hpp"
#include "transformations/snippets/x64/pass/brgemm_to_brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/remove_converts.hpp"
#include "transformations/snippets/x64/pass/enforce_precision.hpp"
#include "transformations/snippets/x64/pass/set_brgemm_cpu_blocking_params.hpp"
#include "transformations/snippets/x64/shape_inference.hpp"
#include "transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp"

#include "shape_inference/custom/subgraph.hpp"
#include "utils/cpu_utils.hpp"
#include "common/primitive_hashing_utils.hpp"

#include "openvino/core/parallel.hpp"


namespace ov {
namespace intel_cpu {
namespace node {

namespace {

struct SnippetKey {
    Subgraph::SnippetAttrs attrs;

    size_t hash() const;
    bool operator==(const SnippetKey& rhs) const;
};

size_t SnippetKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    for (const auto& order : attrs.inMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.inMemPrecs)
        seed = hash_combine(seed, prec.hash());

    for (const auto& order : attrs.outMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.outMemPrecs)
        seed = hash_combine(seed, prec.hash());

    seed = hash_combine(seed, attrs.bodyHash);
    seed = hash_combine(seed, attrs.broadcasting_mask);

    return seed;
}

bool SnippetKey::operator==(const SnippetKey& rhs) const {
    if (attrs.bodyHash != rhs.attrs.bodyHash ||
        attrs.broadcasting_mask != rhs.attrs.broadcasting_mask)
        return false;
    if (attrs.inMemOrders.size() != rhs.attrs.inMemOrders.size() ||
        attrs.inMemPrecs.size() != rhs.attrs.inMemPrecs.size())
        return false;
    if (attrs.outMemOrders.size() != rhs.attrs.outMemOrders.size() ||
        attrs.outMemPrecs.size() != rhs.attrs.outMemPrecs.size())
        return false;

    for (size_t i = 0; i < attrs.inMemOrders.size(); i++) {
        if (!(attrs.inMemOrders[i] == rhs.attrs.inMemOrders[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.outMemOrders.size(); i++) {
        if (!(attrs.outMemOrders[i] == rhs.attrs.outMemOrders[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.inMemPrecs.size(); i++) {
        if (!(attrs.inMemPrecs[i] == rhs.attrs.inMemPrecs[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.outMemPrecs.size(); i++) {
        if (!(attrs.outMemPrecs[i] == rhs.attrs.outMemPrecs[i]))
            return false;
    }

    return true;
}
} // namespace

Subgraph::Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, SnippetShapeInferFactory(op)) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ?
        dnnl::impl::cpu::x64::avx512_core : dnnl::impl::cpu::x64::avx2;
    const auto& tmp_snippet = ov::as_type_ptr<snippets::op::Subgraph>(op);
    OPENVINO_ASSERT(tmp_snippet, "Attempt to create Snippet node from an invalid op type");
    snippetAttrs.snippet = tmp_snippet->clone();
    snippetAttrs.bodyHash = get_body_hash(tmp_snippet);

#if defined(OPENVINO_ARCH_X86_64)
    snippetAttrs.snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
#else
    OPENVINO_THROW("CPU plugin: Snippets code-generator is not supported on non-x64 platforms");

#endif // OPENVINO_ARCH_X86_64

    // Note: we have to update shapeInfer, so it uses the per-thread op::Subgraph copy
    shapeInference = SnippetShapeInferFactory(snippetAttrs.snippet).makeShapeInfer();
    is_dynamic = isDynamicNgraphNode(op);
}

uint64_t Subgraph::get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet) {
    uint64_t seed = 0;
    ov::snippets::pass::Hash hash_function(seed);
    hash_function.run_on_model(snippet->body_ptr());
    return seed;
}

void Subgraph::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::set<ov::element::Type> supportedPrecisions =
        {ov::element::f32, ov::element::i32, ov::element::bf16, ov::element::f16, ov::element::i8, ov::element::u8};

    bool dimRanksAreEqual = true;
    for (size_t i = 0; dimRanksAreEqual && i < inputShapes.size(); i++) {
        for (size_t j = 0; dimRanksAreEqual && j < outputShapes.size(); j++) {
            if (inputShapes[i].getRank() != outputShapes[j].getRank())
                dimRanksAreEqual = false;
        }
    }

    const size_t ndims = outputShapes[0].getRank();
    // Domain sensitive operations support only Planar layout
    const bool isOnlyPlanarApplicable = snippetAttrs.snippet->has_domain_sensitive_ops();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1u, 2u, 3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable;
    // Todo: Snippets currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
    bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable;

    for (const auto& inShape : inputShapes) {
        if (isDynamic && inShape.getRank() != 1)
            isBlockedApplicable = isBlockedApplicable && inShape.getMinDims()[1] != Shape::UNDEFINED_DIM && inShape.getMinDims()[1] > 1;
    }

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };
    auto initDesc = [&] (LayoutType lt) -> NodeDesc {
        auto createMemoryDesc = [lt](const Shape &shape, ov::element::Type prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
            const auto &dims = shape.getDims();
            if (lt == ChannelsFirst && shape.getRank() != 1) {
                auto ndims = shape.getRank();
                VectorDims order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                VectorDims blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else if (lt == Blocked && shape.getRank() != 1 && (shape.getMinDims()[1] != Shape::UNDEFINED_DIM && shape.getMinDims()[1] > 1)) {
                size_t blockSize = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 16 : 8;

                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = dims[1] != Shape::UNDEFINED_DIM ? div_up(blocks[1], blockSize) : Shape::UNDEFINED_DIM;
                blocks.push_back(blockSize);
                order.push_back(1);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else {
                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            }
        };

        size_t offset = 0;
        NodeConfig config;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            const auto originalInputPrecision = getOriginalInputPrecisionAtPort(i);
            const auto precision = ((originalInputPrecision == ov::element::f32) &&
                                     context->getConfig().inferencePrecision == ov::element::bf16 &&
                                     snippetAttrs.snippet->has_domain_sensitive_ops()) ?
                static_cast<ov::element::Type>(ov::element::bf16) :
                originalInputPrecision;
            if (supportedPrecisions.count(precision) == 0)
                OPENVINO_THROW("Subgraph node with name `", getName(), "` doesn't support ", precision, " precision.");

            const auto equalPrecisions = getOriginalOutputPrecisions().size() == 1 &&
                    precision == getOriginalOutputPrecisionAtPort(0);

            BlockedMemoryDesc::CmpMask inputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace((!i && canBeInPlace() && equalPrecisions) ? 0 : -1);
            portConfig.constant(false);
            if (inputShapes[i].getDims()[0] == 1) {
                inputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(inputShapes[i], precision, offset), inputMask);
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            auto precision = getOriginalOutputPrecisionAtPort(i);
            if (supportedPrecisions.count(precision) == 0)
                OPENVINO_THROW("Subgraph node with name `", getName(), "` doesn't support ", precision, " precision.");

            BlockedMemoryDesc::CmpMask outputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            if (outputShapes[i].getDims()[0] == 1) {
                outputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(outputShapes[i], precision, offset), outputMask);
            config.outConfs[i] = portConfig;
        }

        impl_desc_type impl_type = impl_desc_type::unknown;
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }
        return {config, impl_type};
    };

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void Subgraph::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), true);
}

void Subgraph::createPrimitive() {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();
    inputNum = config.inConfs.size();
    outputNum = config.outConfs.size();

    init_memory_ptrs();
    init_attrs();
    init_start_offsets();
    generate();

    Node::createPrimitive();
}

void Subgraph::init_memory_ptrs() {
    srcMemPtrs.resize(inputNum);
    dstMemPtrs.resize(outputNum);
    for (size_t i = 0; i < inputNum; i++)
        srcMemPtrs[i] = getParentEdgeAt(i)->getMemoryPtr();
    for (size_t i = 0; i < outputNum; i++)
        dstMemPtrs[i] = getChildEdgeAt(i)->getMemoryPtr();
}

void Subgraph::init_attrs() {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();

    snippetAttrs.inMemPrecs.resize(inputNum);
    snippetAttrs.outMemPrecs.resize(outputNum);

    snippetAttrs.inMemOrders.resize(inputNum);
    snippetAttrs.outMemOrders.resize(outputNum);

    snippetAttrs.has_non_planar_inputs = false;

    for (size_t i = 0; i < inputNum; i++) {
        const auto& memDesc = config.inConfs[i].getMemDesc();
        snippetAttrs.inMemPrecs[i] = memDesc->getPrecision();
        snippetAttrs.inMemOrders[i] = memDesc->as<BlockedMemoryDesc>()->getOrder();
        snippetAttrs.has_non_planar_inputs |= !memDesc->hasLayoutType(LayoutType::ncsp);
    }
    for (size_t i = 0; i < outputNum; i++) {
        const auto& memDesc = config.outConfs[i].getMemDesc();
        snippetAttrs.outMemPrecs[i] = memDesc->getPrecision();
        snippetAttrs.outMemOrders[i] = memDesc->as<BlockedMemoryDesc>()->getOrder();
    }
}

void Subgraph::init_start_offsets() {
    auto get_offset = [](const BlockedMemoryDescPtr& desc) {
        return static_cast<ptrdiff_t>(desc->getOffsetPadding() * desc->getPrecision().size());
    };
    start_offset_in.resize(inputNum);
    start_offset_out.resize(outputNum);
    for (size_t i = 0; i < inputNum; i++)
        start_offset_in[i] = get_offset(srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>());
    for (size_t i = 0; i < outputNum; i++)
        start_offset_out[i] = get_offset(dstMemPtrs[i]->getDescWithType<BlockedMemoryDesc>());
}

void Subgraph::init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes) {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();

    in_blocked_shapes.reserve(inputNum);
    for (size_t i = 0; i < inputNum; i++) {
        const auto& memDesc = config.inConfs[i].getMemDesc();
        const auto& blockedDesc = memDesc->as<BlockedMemoryDesc>();
        const auto& order = blockedDesc->getOrder();

        in_blocked_shapes.emplace_back(blockedDesc->getBlockDims(), order);
    }
}

void Subgraph::init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types) {
    input_types.reserve(inputNum);
    output_types.reserve(outputNum);
    for (const auto& p : snippetAttrs.inMemPrecs)
        input_types.push_back(p);
    for (const auto& p : snippetAttrs.outMemPrecs)
        output_types.push_back(p);
}

void Subgraph::generate() {
    // here we should perform all shape-agnostic snippets passes
    // * canonicalization (RankNormalization insert)
    // * precision propagation & align element types
    // * data flow optimizations
    // The result of these transformations will be reused by all shapes
    using Manager = snippets::pass::Manager;
    std::vector<Manager::PositionedPass> backend_passes;
#if defined(OPENVINO_ARCH_X86_64)
    using PassPosition = snippets::pass::Manager::PassPosition;
    using Place = snippets::pass::Manager::PassPosition::Place;
#   define SNIPPETS_REGISTER_PASS(PASS_POS, PASS, ...) \
            backend_passes.emplace_back(PASS_POS, std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS(PASS_POS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineStart), ConvertToSwishCPU);
    if (context->getConfig().inferencePrecision == ov::element::bf16 && snippetAttrs.snippet->has_domain_sensitive_ops()) {
        // enforce BF16 precisions to supported operations
        // MatMul has to be decomposed to Brgemm operations before enforcement
        // Note, MatMul decomposition will be run later again for case if BF16 enforcement is not happened
        SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineStart), ov::snippets::pass::MatMulToBrgemm);
        SNIPPETS_REGISTER_PASS(PassPosition(Place::After, "MatMulToBrgemm"), pass::EnforcePrecision, element::f32, element::bf16);
    }

    SNIPPETS_REGISTER_PASS(PassPosition(Place::Before, "PropagatePrecision"), ov::intel_cpu::pass::BrgemmToBrgemmCPU);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::Before, "PropagatePrecision"), ov::intel_cpu::pass::SetBrgemmCPUBlockingParams);

    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineEnd), ov::intel_cpu::pass::RemoveConverts);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineEnd), ov::intel_cpu::pass::MulAddToFMA);

#undef SNIPPETS_REGISTER_PASS

    snippets::op::Subgraph::BlockedShapeVector in_blocked_shapes;
    std::vector<ov::element::Type> input_precisions, output_precisions;
    init_snippets_blocked_shapes(in_blocked_shapes);
    init_precisions(input_precisions, output_precisions);

    snippetAttrs.snippet->data_flow_transformations(in_blocked_shapes, input_precisions, output_precisions, backend_passes);
    snippetAttrs.snippet->convert_body_to_linear_ir(std::make_shared<snippets::CPUShapeInferSnippetsFactory>());

    // todo: snippets don't support backend-provided blocking, so we need to reshape body
    //  using blocked shapes first. This can be removed after [121670]
    // if (snippetAttrs.has_non_planar_inputs) {
    //     std::vector<snippets::VectorDimsRef> in_shapes;
    //     for (const auto& s : snippetAttrs.inMemBlockedDims)
    //         in_shapes.emplace_back(s);
    //     snippetAttrs.snippet->shape_infer(in_shapes);
    // }
    //const VectorDims& canonicalShape = {};

    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensor_rank = rank6D; // std::max(static_cast<size_t>(rank6D), canonicalShape.size());
    snippetAttrs.snippet->set_tensor_rank(tensor_rank);
    snippetAttrs.snippet->set_min_parallel_work_amount(static_cast<size_t>(parallel_get_max_threads()));
    // Note: minimal JIT work amount is a predefined value that describes the number of kernel iterations (work amount)
    // needed to cover kernel call overhead. It is used for balancing between parallel and JIT work amounts in domain optimization.
    snippetAttrs.snippet->set_min_jit_work_amount(256);

    jit_snippets_compile_args jcp;
    jcp.parallel_executor_ndims = tensor_rank;

    ov::snippets::lowered::pass::PassPipeline control_flow_markup_pipeline;
    CPU_REGISTER_PASS_X64(control_flow_markup_pipeline, ov::intel_cpu::pass::BrgemmBlocking)

    ov::snippets::lowered::pass::PassPipeline control_flow_pipeline;
    CPU_REGISTER_PASS_X64(control_flow_pipeline, ov::intel_cpu::pass::FuseLoadStoreConvert)
    CPU_REGISTER_PASS_X64(control_flow_pipeline, ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape);
    schedule = snippetAttrs.snippet->generate_from_linear_ir(control_flow_markup_pipeline,
                                                             control_flow_pipeline,
                                                             reinterpret_cast<const void*>(&jcp));
}

bool Subgraph::needPrepareParams() const {
    return inputShapesModified();
}

void Subgraph::prepareParams() {
    const auto runtime_config = snippetAttrs.snippet->configure();
    loop_args.clear();
    for (const auto& loops : runtime_config.get_loops()) {
        for (const auto& loop : loops.second) {
            loop_args.emplace_back(loop.work_amount, loop.ptr_increments, loop.finalization_offsets);
        }
    }
    data_offsets = runtime_config.get_data_offsets();
}

void Subgraph::update_ptrs(jit_snippets_call_args& call_args,
                           const int64_t indexes[5],
                           const std::vector<MemoryPtr>& inMemPtrs,
                           const std::vector<MemoryPtr>& outMemPtrs,
                           const std::vector<std::vector<int64_t>>& data_offsets) {
    OPENVINO_ASSERT(data_offsets.front().size() == tensor_rank, "Data offsets with invalid ranks detected");
    OPENVINO_ASSERT(data_offsets.size() == inMemPtrs.size() + outMemPtrs.size(), "Incorrect data offset count!");

    for (size_t i = 0; i < inMemPtrs.size(); i++) {
        auto i_ptr = reinterpret_cast<uint8_t*>(inMemPtrs[i]->getData()) + start_offset_in[i];
        for (size_t j = 0; j < tensor_rank - 1; j++) {
            i_ptr += (data_offsets[i][j] * indexes[j]);
        }
        call_args.src_ptrs[i] = i_ptr;
    }
    for (size_t i = 0; i < outMemPtrs.size(); i++) {
        auto i_ptr = reinterpret_cast<uint8_t*>(outMemPtrs[i]->getData()) + start_offset_out[i];
        for (size_t j = 0; j < tensor_rank - 1; j++) {
            i_ptr += (data_offsets[i + inMemPtrs.size()][j] * indexes[j]);
        }
        call_args.dst_ptrs[i] = i_ptr;
    }
    //if (buffer_scratchpad_size > 0) {
    //    call_args.buffer_scratchpad_ptr =
    //            reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    //}
    // todo: remove this assert when jit_snippets_dynamic_call_args are in the final state
    OPENVINO_ASSERT(std::is_standard_layout<jit_snippets_call_args>::value, "JIT dynamic call args are not standard-layout class");
}

void Subgraph::execute(dnnl::stream strm) {
     const auto& dom = snippetAttrs.snippet->get_parallel_exec_domain();
    // < N, C, H, W > < 1, 1, N, C*H*W>
    const auto& callable = schedule.get_callable<dynamic_kernel>();

    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};
            // todo: jit_snippets_call_args are destructed at the end of this lambda.
            //  It means that rather expensive memory allocation-deallocation is performed inside this loop.
            //  A possible solution is to create thread-local jit_snippets_call_args that would be reused here.
            jit_snippets_call_args call_args;
            call_args.register_loops(loop_args);
            update_ptrs(call_args, indexes, srcMemPtrs, dstMemPtrs, data_offsets);
            callable(&call_args);
        });
}

void Subgraph::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

ov::element::Type Subgraph::getRuntimePrecision() const {
    std::vector<ov::element::Type> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated && !parentEdge->getParent()->isConstant()) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToElementType((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}


bool Subgraph::canBeInPlace() const {
    if (isDynamic || getParentEdgesAtPort(0)[0]->getParent()->getType() == Type::Input) {
        return false;
    }
    if (getChildEdges().size() != 1) {
        return false;
    }

    for (auto& parentEdge : getParentEdges()) {
        auto parent = parentEdge.lock()->getParent();
        if (parent->getChildEdges().size() != 1)
            return false;

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Type::Concatenation) {
            for (auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1)
                    return false;
            }
        }
    }
    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

bool Subgraph::created() const {
    return getType() == Type::Subgraph;
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
