/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
#include <cutlass/numeric_types.h>

#define NUM_PROFILE 200 


#define BIT_WIDTH 8
// #define BIT_WIDTH 16
// #define BIT_WIDTH 8

// #define BIT_WIDTH 1

#if BIT_WIDTH == 32
  typedef float input_t;
  typedef float output_t;
#elif BIT_WIDTH == 16
  typedef cutlass::half_t input_t;
  typedef cutlass::half_t output_t;
#elif BIT_WIDTH == 8
  typedef int8_t input_t;
  typedef int32_t output_t;
#elif BIT_WIDTH == 4
  typedef cutlass::int4b_t input_t;
  typedef int32_t output_t;
#elif BIT_WIDTH == 1
  typedef cutlass::uint1b_t input_t;
  typedef int32_t output_t;
#endif


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = output_t;                   // <- data type of accumulator
using ElementComputeEpilogue = output_t;               // <- data type of epilogue operations
using ElementInputA = input_t;                        // <- data type of elements in input matrix A
using ElementInputB = input_t;                        // <- data type of elements in input matrix B
using ElementOutput = output_t;                        // <- data type of elements in output matrix D


// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//-------------full precision CUDA core (PASS) --------------------
#if BIT_WIDTH == 32

using Element = float;

using Gemm = cutlass::gemm::device::Gemm<
  Element, 
  cutlass::layout::RowMajor,
  Element, 
  cutlass::layout::ColumnMajor,
  Element,
  cutlass::layout::RowMajor, 
  Element,
  cutlass::arch::OpClassSimt, 
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<32, 64, 8>,
  cutlass::gemm::GemmShape<32, 64, 8>, 
  cutlass::gemm::GemmShape<1, 1, 1>,
  cutlass::epilogue::thread::LinearCombination<
      Element, 
      1,
      Element, 
      Element>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
  4
>;


//-------------half precision Tensor core (PASS) --------------------
#elif BIT_WIDTH == 16

using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<64, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 8>,
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    64 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

//-------------INT-8 Tensor core (PASS) --------------------
#elif BIT_WIDTH == 8

using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

using Gemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor, 
    int8_t, cutlass::layout::ColumnMajor, 
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>, 
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationClamp<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;


//-------------INT-4 Tensor core (PASS) --------------------
#elif BIT_WIDTH == 4

using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::int4b_t,
  cutlass::layout::RowMajor,
  cutlass::int4b_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm75,
  cutlass::gemm::GemmShape<128, 256, 128>,
  cutlass::gemm::GemmShape<64, 64, 128>,
  cutlass::gemm::GemmShape<8, 8, 32>,
  cutlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

//-------------INT-1 Tensor core (PASS)--------------------
#elif BIT_WIDTH == 1
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;
    const int pipe_stages = 4;

    using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor, 
    cutlass::uint1b_t, cutlass::layout::ColumnMajor, 
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    // RTX3090 setting for block, warp, and mma shape
    cutlass::gemm::GemmShape<128, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>, 
    cutlass::gemm::GemmShape<8, 8, 128>,
    // A100 setting for block, warp, and mma shape
    // cutlass::gemm::GemmShape<256, 128, 1024>,
    // cutlass::gemm::GemmShape<64, 64, 1024>, 
    // cutlass::gemm::GemmShape<16, 8, 256>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, pipe_stages, 128, 128,
    false, cutlass::arch::OpXorPopc>;
    
#endif


void cut_gemm(input_t *A, input_t* B, output_t* C,int rowA,int colA, int rowB,int colB);

