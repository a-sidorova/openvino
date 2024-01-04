// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#include "emitters/snippets/x64/jit_kernel_emitter.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Subgraph : public Node {
    class SnippetExecutor;
    class SnippetJitExecutor;
    class SnippetJitStaticExecutor;
    class SnippetJitShapeAgnosticExecutor;
    class SnippetJitDynamicSpecializedExecutor;

public:
    Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    ~Subgraph() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    ov::element::Type getRuntimePrecision() const override;

    void createPrimitive() override;
    void prepareParams() override;

    IShapeInfer::Result shapeInfer() const override;

    bool canBeInPlace() const override;
    bool created() const override;

    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    struct SnippetAttrs {
        // Local copy of subgraph node for canonization & code generation
        std::shared_ptr<snippets::op::Subgraph> snippet;
        uint64_t bodyHash;
        std::vector<VectorDims> inMemOrders;
        std::vector<VectorDims> outMemOrders;
        std::vector<ov::element::Type> inMemPrecs;
        std::vector<ov::element::Type> outMemPrecs;
        // todo: used flag if we need extra shape infer, can be removed after [121670]
        bool has_non_planar_inputs;
    };

private:
    static uint64_t get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet);

    void init_memory_ptrs();
    void init_attrs();
    void init_start_offsets();
    void init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes) const;
    void init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types) const;
    void init_blocked_broadcasting_mask(uint8_t& mask) const;
    void lower();

    std::vector<ov::snippets::pass::Manager::PositionedPassBase> get_data_flow_passes() const;
    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> get_control_flow_passes() const;

    uint8_t get_broadcasting_mask(const std::vector<VectorDims>& input_shapes) const;
    // Plugin static shapes
    std::vector<VectorDims> get_input_blocked_shapes() const;
    std::vector<VectorDims> get_output_planar_shapes() const;

    std::shared_ptr<SnippetAttrs> snippetAttrs;
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;

    size_t input_num = 0;
    size_t output_num = 0;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    size_t tensor_rank = 0;
    bool is_dynamic = false;

    static const size_t rank6D {6};

    mutable std::shared_ptr<SnippetExecutor> execPtr = nullptr;
};

class Subgraph::SnippetExecutor {
public:
    SnippetExecutor(const std::shared_ptr<SnippetAttrs>& attrs, size_t tensor_rank,
                    const std::vector<ptrdiff_t>& start_offset_in,
                    const std::vector<ptrdiff_t>& start_offset_out);
    virtual ~SnippetExecutor() = default;

    virtual void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;

protected:
    std::shared_ptr<SnippetAttrs> snippet_attrs;
    size_t tensor_rank = 0;

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};
};

// Base class for Jit Executors
class Subgraph::SnippetJitExecutor : public Subgraph::SnippetExecutor {
public:
    SnippetJitExecutor(const std::shared_ptr<SnippetAttrs>& attrs, size_t tensor_rank,
                       const std::vector<ptrdiff_t>& start_offset_in,
                       const std::vector<ptrdiff_t>& start_offset_out);
    virtual ~SnippetJitExecutor() = default;

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

protected:
    // Evaluates generated snippet using parallel backend
    virtual void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
    virtual void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;

    void init_runtime_params();
    void generate();

    std::shared_ptr<snippets::Schedule> schedule;
    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> parallel_exec_domain = {};
    size_t harness_work_amount = 0;

    // Buffer scratchpad
    std::vector<uint8_t> buffer_scratchpad = {};
    size_t buffer_scratchpad_size = 0;
};

// Class for Subgraphs with static shapes
class Subgraph::SnippetJitStaticExecutor : public Subgraph::SnippetJitExecutor {
public:
    SnippetJitStaticExecutor(const std::shared_ptr<SnippetAttrs>& attrs, size_t tensor_rank,
                             const std::vector<ptrdiff_t>& start_offset_in,
                             const std::vector<ptrdiff_t>& start_offset_out);

protected:
    typedef void (*kernel)(const void*, const void*);

    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs);
};

// Class for dynamic Subgraph with shape-agnostic that just generate lowered code
class Subgraph::SnippetJitShapeAgnosticExecutor : public Subgraph::SnippetJitExecutor {
    friend class Subgraph::SnippetJitDynamicSpecializedExecutor;
public:
    SnippetJitShapeAgnosticExecutor(const std::shared_ptr<SnippetAttrs>& attrs, size_t tensor_rank,
                                    const std::vector<ptrdiff_t>& start_offset_in,
                                    const std::vector<ptrdiff_t>& start_offset_out);

protected:
    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

// Specialized dynamic executor based on shape agnostic kernel for the specific input shapes
class Subgraph::SnippetJitDynamicSpecializedExecutor : public Subgraph::SnippetJitExecutor {
public:
    SnippetJitDynamicSpecializedExecutor(const std::shared_ptr<SnippetJitShapeAgnosticExecutor>& agnostic,
                                         const std::vector<VectorDims>& input_blocked_shapes,
                                         const std::vector<VectorDims>& output_planar_shapes);

    const std::vector<VectorDims>& get_output_shapes() const { return output_shapes; }

protected:
    typedef void (*dynamic_kernel)(const void *);

    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs,
                            const int64_t* indexes);
    // Evaluates generated snippet using parallel backend
    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

    std::vector<std::vector<size_t>> data_offsets = {};
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};

    std::vector<VectorDims> output_shapes;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
