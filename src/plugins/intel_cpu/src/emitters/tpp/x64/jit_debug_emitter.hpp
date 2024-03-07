// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "jit_tpp_emitter.hpp"
#include "jit_eltwise_emitters.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @interface DebugTppEmitter
 * @brief The purpose of this emitter is to facilitate debugging of TPP emitters. It allows to access attributes of the
 * owned Tpp emitter at runtime, inspect source ov::Node, in/out memory before and after execution, etc.
 */
class DebugTppEmitter : public TppEmitter {
public:
    DebugTppEmitter(const ov::snippets::lowered::ExpressionPtr& expr, const std::shared_ptr<TppEmitter>& original)
            : TppEmitter(*original),
            m_original(original),
            m_compiled_kernel(m_original->get_compiled_kernel_ptr()),
            m_execute_function(m_original->get_execute_function_ptr()),
            m_source_node(expr->get_node()) {
    }

    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override {
        m_original->validate_arguments(in, out);
    };

    size_t get_inputs_num() const override { return num_kernel_args - 1; }

protected:
    static void execute_kernel_unary(const DebugTppEmitter* emitter, void *in0, void *out0) {
        OV_CPU_JIT_EMITTER_ASSERT(emitter && emitter->m_execute_function && emitter->m_compiled_kernel,
                                  "Unable to execute unary kernel");
        // Note: put a breakpoint here and analyze all the necessary debug info in runtime
        const auto original = std::dynamic_pointer_cast<UnaryEltwiseTppEmitter>(emitter->m_original);
        OV_CPU_JIT_EMITTER_ASSERT(original, "Incorrect emitter");
        const auto& m_shape = original->m_shape;
        float* in_ptr = reinterpret_cast<float*>(in0);
        float* out_ptr = reinterpret_cast<float*>(out0);
        for (int n = 0; n < m_shape.n; n++) {
            mempcpy(out_ptr, in_ptr, m_shape.m * sizeof(float));
            in_ptr += m_shape.ldi;
            out_ptr += m_shape.ldo;
        }
    }

    static void execute_kernel_binary(const DebugTppEmitter* emitter, void* in0, void* in1, void* out0) {
        OV_CPU_JIT_EMITTER_ASSERT(emitter && emitter->m_execute_function && emitter->m_compiled_kernel,
                                  "Unable to execute binary kernel");
        // Note: put a breakpoint here and analyze all the necessary debug info in runtime
        const auto original = std::dynamic_pointer_cast<BinaryEltwiseTppEmitter>(emitter->m_original);
        OV_CPU_JIT_EMITTER_ASSERT(original, "Incorrect emitter");
        const auto& m_shape = original->m_shape;
        const auto& m_flags = original->m_compile_flags;
        const auto& ldi = original->is_first ? m_shape.ldi : m_shape.ldi2;
        float* in_ptr = original->is_first ? reinterpret_cast<float*>(in0) : reinterpret_cast<float*>(in1);
        float* out_ptr = reinterpret_cast<float*>(out0);
        for (int n = 0; n < m_shape.n; n++) {
            mempcpy(out_ptr, in_ptr, m_shape.m * sizeof(float));
            in_ptr += ldi;
            out_ptr += m_shape.ldo;
        }
    }

    const uintptr_t get_execute_function_ptr() const override {
        // Note: num_kernel_args accounts for both input and output args
        switch (num_kernel_args) {
            case 2: return reinterpret_cast<const uintptr_t>(execute_kernel_unary);
            case 3: return reinterpret_cast<const uintptr_t>(execute_kernel_binary);
            default: OV_CPU_JIT_EMITTER_THROW("More than two arguments are not supported");
        }
    }

    const uintptr_t get_compiled_kernel_ptr() const override {
        return reinterpret_cast<const uintptr_t>(this);
    }

private:
    std::shared_ptr<TppEmitter> m_original {nullptr};
    uintptr_t m_compiled_kernel {0};
    uintptr_t m_execute_function {0};
    std::shared_ptr<ov::Node> m_source_node {nullptr};
};

}   // namespace intel_cpu
}   // namespace ov
