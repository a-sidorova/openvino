# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            MKLDNNPlugin
        LINK_LIBRARIES
            funcSharedTests
        ADD_CPPLINT
        LABELS
            CPU
)

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::mish_compute_vector(
        const Vmm &vmm_src) {
    // Save src data on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // ln(1+exp(x))
    soft_relu_compute_vector(vmm_src);
    // tanh(ln(1+exp(x)))
    tanh_compute_vector(vmm_src);
    // x*tanh(ln(1+exp(x)))
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}
