/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <iostream>
#include <chrono>
#include <random>
#include <mateval/mateval.hpp>
#include <cutf/cublas.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
        cutlass::gemm::GemmShape<128, 64, 32>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
        // memory access. For a byte, it's 16
        // elements. This becomes the vector width of
        // math instructions in the epilogue too
        ElementAccumulator,                                // <- data type of accumulator
        ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ShapeMMAThreadBlock,
        ShapeMMAWarp,
        ShapeMMAOp,
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages>;

void run() {
    auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
    std::printf(
            "m,n,k,tm,tn,tk,rm,rn,rk,fm,fn,fk,stage,cutlass_time,cutlass_performance,cutlass_residual,cutlass_max_error,cublas_time,cublas_performance,cublas_residual,cublas_max_error\n");
    for (unsigned i = 10; i < 25; i++) {
        const int length_m = 1 << 11;
        const int length_n = 1 << 11;
        const int length_k = 1 << i;

        // Create a tuple of problem size for matrix multiplication
        cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

        // Initialize tensors using CUTLASS helper functions
        cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
                problem_size.mk());  // <- Create matrix A with dimensions M x K
        cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
                problem_size.kn());  // <- Create matrix B with dimensions K x N
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
                problem_size.mn());  // <- Create matrix C with dimensions M x N
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
                problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
        // CUTLASS kernel
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
                problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
        // reference kernel

        std::mt19937 mt(std::random_device{}());
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (unsigned i = 0; i < length_m * length_k; i++) {
            tensor_a.host_data()[i] = dist(mt);
        }
        for (unsigned i = 0; i < length_n * length_k; i++) {
            tensor_b.host_data()[i] = dist(mt);
        }
        for (unsigned i = 0; i < length_n * length_m; i++) {
            tensor_c.host_data()[i] = dist(mt);
        }

        // Copy data from host to GPU
        tensor_a.sync_device();
        tensor_b.sync_device();
        tensor_c.sync_device();
        tensor_d.sync_device();
        tensor_ref_d.sync_device();

        // Initialize alpha and beta for dot product computation
        ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
        ElementComputeEpilogue beta = ElementComputeEpilogue(0);

        // Split K dimension into 1 partitions
        int split_k_slices = 1;

        // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
        // instantiated CUTLASS kernel
        typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                           tensor_a.device_ref(),  // <- reference to matrix A on device
                                           tensor_b.device_ref(),  // <- reference to matrix B on device
                                           tensor_c.device_ref(),  // <- reference to matrix C on device
                                           tensor_d.device_ref(),  // <- reference to matrix D on device
                                           {alpha, beta},          // <- tuple of alpha and beta
                                           split_k_slices};        // <- k-dimension split factor

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm_op;

        // Check the problem size is supported or not
        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_CHECK(status);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        // Launch initialized CUTLASS kernel
        cudaDeviceSynchronize();
        const auto cutlass_start_clock = std::chrono::high_resolution_clock::now();
        status = gemm_op();
        cudaDeviceSynchronize();
        const auto cutlass_end_clock = std::chrono::high_resolution_clock::now();
        CUTLASS_CHECK(status);

        const auto cutlass_elapsed_time =
                std::chrono::duration_cast<std::chrono::microseconds>(cutlass_end_clock - cutlass_start_clock).count() *
                1e-6;
        const auto cutlass_performance = 2. / cutlass_elapsed_time * length_k * length_m * length_n / (1lu << 40);

        // Create instantiation for device reference gemm kernel
        cutlass::reference::device::Gemm<ElementInputA,
                LayoutInputA,
                ElementInputB,
                LayoutInputB,
                ElementOutput,
                LayoutOutput,
                ElementComputeEpilogue,
                ElementComputeEpilogue>
                gemm_device;

        // Launch device reference gemm kernel
        gemm_device(problem_size,
                    alpha,
                    tensor_a.device_ref(),
                    tensor_b.device_ref(),
                    beta,
                    tensor_c.device_ref(),
                    tensor_ref_d.device_ref());

        // Wait for kernels to finish
        cudaDeviceSynchronize();

        // Copy output data from CUTLASS and reference kernel to host for comparison
        tensor_d.sync_host();
        tensor_ref_d.sync_host();

        const auto cutlass_residual = mtk::mateval::residual(
                length_m, length_n,
                mtk::mateval::col_major, mtk::mateval::col_major,
                tensor_d.host_data(), length_m,
                tensor_ref_d.host_data(), length_m
        );
        const auto cutlass_max_error = mtk::mateval::max_error(
                length_m, length_n,
                mtk::mateval::col_major, mtk::mateval::col_major,
                tensor_d.host_data(), length_m,
                tensor_ref_d.host_data(), length_m
        );


        cudaDeviceSynchronize();
        const auto cublas_start_clock = std::chrono::high_resolution_clock::now();
        cutf::cublas::gemm(
                *cublas_handle.get(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                length_m, length_n, length_k,
                &alpha,
                tensor_a.device_data(), length_k,
                tensor_b.device_data(), length_k,
                &beta,
                tensor_c.device_data(), length_m
        );
        cudaDeviceSynchronize();
        const auto cublas_end_clock = std::chrono::high_resolution_clock::now();

        const auto cublas_elapsed_time =
                std::chrono::duration_cast<std::chrono::microseconds>(cublas_end_clock - cublas_start_clock).count() *
                1e-6;
        const auto cublas_performance = 2. / cublas_elapsed_time * length_k * length_m * length_n / (1lu << 40);

        tensor_c.sync_host();
        const auto cublas_residual = mtk::mateval::residual(
                length_m, length_n,
                mtk::mateval::col_major, mtk::mateval::row_major,
                tensor_c.host_data(), length_m,
                tensor_ref_d.host_data(), length_m
        );
        const auto cublas_max_error = mtk::mateval::max_error(
                length_m, length_n,
                mtk::mateval::col_major, mtk::mateval::row_major,
                tensor_c.host_data(), length_m,
                tensor_ref_d.host_data(), length_m
        );
#ifdef TC_COR
        std::printf("cor,");
#else
        std::printf("nocor,");
#endif
        std::printf("%d,%d,%d,",
                    length_m, length_n, length_k
        );

        std::printf("%d,%d,%d,%lu,",
                    ShapeMMAThreadBlock::kM,
                    ShapeMMAThreadBlock::kN,
                    ShapeMMAThreadBlock::kK,
                    (ShapeMMAThreadBlock::kM * ShapeMMAThreadBlock::kN +
                     ShapeMMAThreadBlock::kM * ShapeMMAThreadBlock::kK +
                     ShapeMMAThreadBlock::kN * ShapeMMAThreadBlock::kK) * sizeof(float)
        );
        std::printf("%d,%d,%d,",
                    ShapeMMAWarp::kM,
                    ShapeMMAWarp::kN,
                    ShapeMMAWarp::kK
        );
        std::printf("%d,%d,%d,",
                    ShapeMMAOp::kM,
                    ShapeMMAOp::kN,
                    ShapeMMAOp::kK
        );
        std::printf("%d,", NumStages);
        std::printf("%e,%e,",
                    cutlass_elapsed_time, cutlass_performance
        );
        std::printf("%e,%e,",
                    cutlass_residual, cutlass_max_error
        );
        std::printf("%e,%e,",
                    cublas_elapsed_time, cublas_performance
        );
        std::printf("%e,%e\n",
                    cublas_residual, cublas_max_error
        );
        std::fflush(stdout);
    }
}

int main() {

    bool notSupported = false;

    // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (!((props.major * 10 + props.minor) >= 80)) {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
                  << std::endl;
        notSupported = true;
    }

    if (notSupported) {
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }
    run();

    return 0;
}
