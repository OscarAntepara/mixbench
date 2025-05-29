cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mixbench-cpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#include_directories("../../include")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE(STATUS "GNU C++ compiler is used")
else()
  MESSAGE(WARNING "Not a GNU C++ compiler is used. Potentially not optimized for this.")
endif()

find_package(OpenMP)
find_package(CUDA REQUIRED)

option(BASELINE_IMPL "Use baseline implementation" OFF)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

if (NOT CMAKE_CXX_FLAGS)
    message (STATUS "CMAKE_CXX_FLAGS not set. Using -march=native")
    set (CMAKE_CXX_FLAGS "-march=native -fopenmp")
endif()

# Get version info from git tag
execute_process(COMMAND git describe --tags --always
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "")
endif()
string(STRIP "${GIT_REV}" GIT_REV)
file(WRITE "version_info.h" "#define VERSION_INFO \"")
file(APPEND "version_info.h" ${GIT_REV})
file(APPEND "version_info.h" "\"")

set(CMAKE_BUILD_TYPE Release)

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
string(APPEND CMAKE_CUDA_FLAGS " --cudart=static")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=[sm_80,compute_80] -gencode arch=compute_80,code=compute_80")


include_directories("../../include")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#add_executable(mixbench-cpu main-cuda.cpp mix_kernels_cuda.h lcutil.h version_info.h mix_kernels_cuda.cu mix_kernels_cuda.h lcutil.h)
add_executable(mixbench-cpu main.cpp mix_kernels_cpu.h lcutil.h mix_kernels_cpu.cu mix_kernels_cpu.h version_info.h lcutil.h)

if(${BASELINE_IMPL})
    message(WARNING "Enforcing baseline implementation")
    target_compile_definitions(mixbench-cpu PRIVATE BASELINE_IMPL)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(mixbench-cpu PUBLIC ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
    target_compile_features(mixbench-cpu PUBLIC cxx_std_11)
endif()

