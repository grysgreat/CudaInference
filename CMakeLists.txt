cmake_minimum_required(VERSION 3.2.2)
project(cuda_proj)

find_package(CUDA REQUIRED)
#find_library(CUDNN_LIB cudnn)

# Set it by yourself, if your cuDNN installation path
# is different from CUDA installation path
#set(CUDNN_PATH ${CUDA_TOOLKIT_ROOT_DIR})

include(FindCUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -DEIGEN_NO_DEBUG")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(Eigen3 REQUIRED)

list(APPEND CUDA_NVCC_FLAGS
    "-arch=sm_75;
    -std=c++17;
    -cudart=shared;"
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0  -fopenmp -g -DDEBUG")
    list(APPEND CUDA_NVCC_FLAGS
        "-g
        -O0"
    )

else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fopenmp -Wall")
    list(APPEND CUDA_NVCC_FLAGS
        "-O3;"
    )
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/cutlass/tools/util/include")
    message(FATAL_ERROR [[
        CUTLASS library not found in third-party/cutlass.
        Please fetch with git submodule update --init --recursive
        ]])
endif()


include_directories(
  "rsvd/include"
    ${Eigen3_INCLUDE_DIR}
)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/cutlass/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/cutlass/tools/util/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/cutlass/examples/common)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

#include_directories(${CUDNN_PATH} ${CUDNN_PATH}/include)
#link_directories(${CUDNN_PATH} ${CUDNN_PATH}/lib ${CUDNN_PATH}/lib64)

set(SOURCE_DIR source)
set(INCLUDE_DIR
    include
)

file(GLOB SOURCES_LIB
    "${SOURCE_DIR}/*.cu"
    "${SOURCE_DIR}/*.cpp"
    "${SOURCE_DIR}/*.c"
)
list(REMOVE_ITEM SOURCES_LIB
    ${SOURCE_DIR}/main.cpp
)

file(GLOB INCLUDE
    "${INCLUDE_DIR}/*.cuh"
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.hpp"
)



include_directories(${INCLUDE_DIR})




cuda_add_library(cuda_proj_lib ${SOURCES_LIB})
cuda_add_executable(cuda_proj ${SOURCE_DIR}/main.cpp)
target_link_libraries(cuda_proj cuda_proj_lib)
cuda_add_cublas_to_target(cuda_proj)

set_target_properties(cuda_proj PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
)
