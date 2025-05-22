cmake_minimum_required(VERSION 3.9)

if(NOT DEFINED ENV{HIP_PATH})
    set(ENV{HIP_PATH} "/opt/rocm/hip")
endif()
if(NOT DEFINED ENV{ROCM_PATH})
    set(ENV{ROCM_PATH} "/opt/rocm/")
endif()
if(NOT DEFINED HIP_PATH)
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
endif()
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCM installation")
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" "${ROCM_PATH}/lib/cmake" {CMAKE_MODULE_PATH})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mixbench-cpu LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
include_directories("../../include")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE(STATUS "GNU C++ compiler is used")
else()
  MESSAGE(WARNING "Not a GNU C++ compiler is used. Potentially not optimized for this.")
endif()

find_package(OpenMP)

option(BASELINE_IMPL "Use baseline implementation" OFF)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

if (NOT CMAKE_CXX_FLAGS)
    message (STATUS "CMAKE_CXX_FLAGS not set. Using -march=native -funroll-loops")
    set (CMAKE_CXX_FLAGS "-march=native -funroll-loops -fopenmp")
endif()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

find_package(HIP)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
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

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

include_directories("../../include")
include_directories("/opt/rocm/include/roctracer/")
link_directories("/shared/apps/rhel8/opt/rocm-6.3.2/lib/libroctx64.so")

set(BENCH_SOURCE_FILES main.cpp mix_kernels_cpu.cpp)

set_source_files_properties(${BENCH_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)


add_executable(mixbench-cpu ${BENCH_SOURCE_FILES})

if(${BASELINE_IMPL})
    message(WARNING "Enforcing baseline implementation")
    target_compile_definitions(mixbench-cpu PRIVATE BASELINE_IMPL)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(mixbench-cpu PUBLIC OpenMP::OpenMP_CXX /shared/apps/rhel8/opt/rocm-6.3.2/lib/libroctx64.so)
endif()

