# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# NOTE: use T-Head compiler:
#    git clone https://github.com/T-head-Semi/xuantie-gnu-toolchain.git
#    ./configure --prefix=/opt/riscv
#    make linux
#    -DRISCV_TOOLCHAIN_ROOT=/opt/riscv

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV64_THEAD ON)
set(RISCV64_RVV0p7 ON)

set(RISCV_TOOLCHAIN_ROOT $ENV{RISCV_TOOLCHAIN_ROOT} CACHE PATH "Path to CLANG for RISC-V cross compiler build directory")
set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot" CACHE PATH "RISC-V sysroot")

set(CMAKE_C_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++)
set(CMAKE_STRIP ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-strip)
set(PKG_CONFIG_EXECUTABLE "NOT-FOUND" CACHE PATH "Path to RISC-V pkg-config")

set(CMAKE_FIND_ROOT_PATH "${RISCV_TOOLCHAIN_ROOT}/riscv64-unknown-linux-gnu")

# Don't run the linker on compiler check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d")
