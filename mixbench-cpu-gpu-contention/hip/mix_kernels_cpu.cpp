/**
 * mix_kernels_cpu.cpp: This file is part of the mixbench GPU micro-benchmark
 *suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <iostream>
#ifdef __CUDACC__
#include <math_constants.h>
#define GPU_INF(_T) (_T)(CUDART_INF)
#else
#include <limits>
#define GPU_INF(_T) std::numeric_limits<_T>::infinity()
#endif

typedef __half2 half2;

#include <common.h>
#include "lhiputil.h"

// roctx header file
#include "/shared/apps/rhel8/opt/rocm-6.3.2/include/roctracer/roctx.h"

#define ELEMENTS_PER_THREAD (8)
#define CPU_COMPUTE_ITERS_CONTENTION (2)


const auto base_omp_get_max_threads = omp_get_max_threads();

using benchmark_clock = std::chrono::steady_clock;

#ifdef BASELINE_IMPL

template <typename Element, size_t compute_iterations, size_t static_chunk_size>
Element __attribute__((noinline)) bench_block(Element* data) {
  Element sum = 0;
  Element f = data[0];

#pragma omp simd aligned(data : 64) reduction(+ : sum)
  for (size_t i = 0; i < static_chunk_size; i++) {
    Element t = data[i];
    for (size_t j = 0; j < compute_iterations; j++) {
      t = t * t + f;
    }
    sum += t;
  }
  return sum;
}

#else

template <typename Element, size_t compute_iterations, size_t static_chunk_size>
Element __attribute__((noinline)) bench_block(Element* data) {
  Element sum = 0;

  Element f[] = {data[0], data[1], data[2], data[3],
                 data[4], data[5], data[6], data[7]};

#pragma omp simd aligned(data : 64) reduction(+ : sum)
  for (size_t i = 0; i < static_chunk_size; i++) {
    Element t[] = {data[i], data[i], data[i], data[i],
                   data[i], data[i], data[i], data[i]};
    for (size_t j = 0; j < compute_iterations / 8; j++) {
      t[0] = t[0] * t[0] + f[0];
      t[1] = t[1] * t[1] + f[1];
      t[2] = t[2] * t[2] + f[2];
      t[3] = t[3] * t[3] + f[3];
      t[4] = t[4] * t[4] + f[4];
      t[5] = t[5] * t[5] + f[5];
      t[6] = t[6] * t[6] + f[6];
      t[7] = t[7] * t[7] + f[7];
    }
    if constexpr (compute_iterations % 8 > 0) {
      t[0] = t[0] * t[0] + f[0];
    }
    if constexpr (compute_iterations % 8 > 1) {
      t[1] = t[1] * t[1] + f[1];
    }
    if constexpr (compute_iterations % 8 > 2) {
      t[2] = t[2] * t[2] + f[2];
    }
    if constexpr (compute_iterations % 8 > 3) {
      t[3] = t[3] * t[3] + f[3];
    }
    if constexpr (compute_iterations % 8 > 4) {
      t[4] = t[4] * t[4] + f[4];
    }
    if constexpr (compute_iterations % 8 > 5) {
      t[5] = t[5] * t[5] + f[5];
    }
    if constexpr (compute_iterations % 8 > 6) {
      t[6] = t[6] * t[6] + f[6];
    }
    sum += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
  }
  return sum;
}

#endif

template <typename Op>
auto measure_operation(Op op) {
  auto timer_start = benchmark_clock::now();
  op();
  auto timer_duration = benchmark_clock::now() - timer_start;
  return std::chrono::duration_cast<std::chrono::microseconds>(timer_duration)
             .count() /
         1000.;
}

template <typename Op>
auto benchmark_max_omp(Op op) {
  constexpr int total_runs = 20;

  auto duration = op();  // drop first measurement
  std::vector<decltype(duration)> measurements;
  // 1st try with full threading
  omp_set_num_threads(base_omp_get_max_threads);

  for (int i = 0; i < total_runs; i++) {
    duration = op();
    measurements.push_back(duration);
  }

  return *std::max_element(std::begin(measurements), std::end(measurements));
}



template <typename Op>
auto benchmark_omp(Op op) {
  constexpr int total_runs = 20;
  constexpr int total_half_thread_runs = 20;

  auto duration = op();  // drop first measurement
  std::vector<decltype(duration)> measurements;

  // 1st try with full threading
  omp_set_num_threads(base_omp_get_max_threads);

  for (int i = 0; i < total_runs; i++) {
    duration = op();
    measurements.push_back(duration);
  }
/*
  // then try with half threading
  if (base_omp_get_max_threads > 1) {
    omp_set_num_threads(base_omp_get_max_threads / 2);

    for (int i = 1; i < total_half_thread_runs; i++) {
      duration = op();
      measurements.push_back(duration);
    }
  }
*/
  return *std::min_element(std::begin(measurements), std::end(measurements));
}

template <class T>
inline __device__ T mad(const T& a, const T& b, const T& c) {
  return a * b + c;
}

template <>
inline __device__ double mad(const double& a,
                             const double& b,
                             const double& c) {
  return fma(a, b, c);
}

template <class T>
inline __device__ bool is_equal(const T& a, const T& b) {
  return a == b;
}


template <class T,
          int blockSize,
          unsigned int granularity,
          unsigned int compute_iterations>
__global__ void benchmark_func(T seed, T* g_data) {
  const int stride = blockSize;
  const int idx = hipBlockIdx_x * blockSize * granularity + hipThreadIdx_x;

  T tmps[granularity];
#pragma unroll
  for (int j = 0; j < granularity; j++) {
    // Load elements (memory intensive part)
    tmps[j] = g_data[idx + j * stride];
    // Perform computations (compute intensive part)
    for (int i = 0; i < compute_iterations; i++) {
      tmps[j] = mad<T>(tmps[j], tmps[j], seed);
    }
  }
  // Multiply add reduction
  T sum = static_cast<T>(0);
#pragma unroll
  for (int j = 0; j < granularity; j += 2) {
    sum = mad<T>(tmps[j], tmps[j + 1], sum);
  }
  // Dummy code
  if (is_equal(sum, static_cast<T>(-1)))  // Designed so it never executes
    g_data[idx] = sum;

}

void initializeEvents_ext(hipEvent_t* start, hipEvent_t* stop) {
  HIP_SAFE_CALL(hipEventCreate(start));
  HIP_SAFE_CALL(hipEventCreate(stop));
}

float finalizeEvents_ext(hipEvent_t start, hipEvent_t stop) {
  HIP_SAFE_CALL(hipGetLastError());
  HIP_SAFE_CALL(hipEventSynchronize(stop));
  float kernel_time;
  HIP_SAFE_CALL(hipEventElapsedTime(&kernel_time, start, stop));
  HIP_SAFE_CALL(hipEventDestroy(start));
  HIP_SAFE_CALL(hipEventDestroy(stop));
  return kernel_time;
}



template <typename Element, size_t compute_iterations>
__attribute__((optimize("unroll-loops"))) size_t bench(size_t len,
                                                       const Element seed1,
                                                       const Element seed2,
                                                       Element* src) {
  Element sum = 0;
  constexpr size_t static_chunk_size = 4096;

#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (size_t it_base = 0; it_base < len; it_base += static_chunk_size) {
    sum += bench_block<Element, compute_iterations, static_chunk_size>(
        &src[it_base]);
  }

  *src = sum;
  return len;
}


void runbench_warmup(double* cd, long size) {
  const long compute_grid_size = size / ELEMENTS_PER_THREAD;
  const int BLOCK_SIZE = 256;
  const int TOTAL_BLOCKS = compute_grid_size / BLOCK_SIZE;
  const long long computations =
      ELEMENTS_PER_THREAD * (long long)compute_grid_size +
      (2 * ELEMENTS_PER_THREAD * 2) *
          (long long)compute_grid_size;
  const long long memoryoperations = size;

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
  hipEvent_t start[2], stop[2];

  constexpr auto total_bench_iterations = 6;
  auto kernel_time_mad_dp_gpu = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start[0], &stop[0]);

    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<double, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       2>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start[0], stop[0], 0, 1.0f, cd);
    return finalizeEvents_ext(start[0], stop[0]);
  });
}


class ComputeSpace {
  size_t memory_space_{0};
  int compute_iterations_{0};

 public:
  ComputeSpace(size_t memory_space, int compute_iterations)
      : memory_space_{memory_space}, compute_iterations_{compute_iterations} {}

  template <typename T>
  size_t compute_ops() const {
    const auto total_elements = element_count<T>();
    const long long computations =
        total_elements            /* Vector length */
            * compute_iterations_ /* Core loop iteration count */
            * 2                   /* Flops per core loop iteration */
            * 1                   /* FMAs in the inner most loop */
        + total_elements - 1      /* Due to sum reduction */
        ;
    return computations;
  }

  size_t memory_traffic() const { return memory_space_; }

  template <typename T>
  size_t element_count() const {
    return memory_space_ / sizeof(T);
  }
};

template <unsigned int compute_iterations>
void runbench(double* c, size_t size) {
  ComputeSpace cs{size * sizeof(double), compute_iterations};
  // floating point part (double prec)
  auto kernel_time_mad_dp = benchmark_omp([&] {
    return measure_operation([&] {
      bench<double, compute_iterations>(cs.element_count<double>(), 1., -1., c);
    });
  });
  const auto computations_dp = cs.compute_ops<double>();
  const auto memory_traffic = cs.memory_traffic();

  printf(
      "CPU,     %4d,   %8.3f, %8.2f, %8.2f, %7.2f\n",
      compute_iterations,
        // DP
       static_cast<double>(computations_dp) / static_cast<double>(memory_traffic),
       kernel_time_mad_dp,
       static_cast<double>(computations_dp) / kernel_time_mad_dp * 1000. /
           static_cast<double>(1000 * 1000 * 1000),
       static_cast<double>(memory_traffic) / kernel_time_mad_dp * 1000. /
          (1000. * 1000. * 1000.)
         );

}

template <unsigned int compute_iterations_cpu, unsigned int compute_iterations_gpu>
void runbench_iters_gpu(double* cd, double* c, long size, int& num_iters) {
  const long compute_grid_size = size / ELEMENTS_PER_THREAD;
  const int BLOCK_SIZE = 256;
  const int TOTAL_BLOCKS = compute_grid_size / BLOCK_SIZE;
  const long long computations =
      ELEMENTS_PER_THREAD * (long long)compute_grid_size +
      (2 * ELEMENTS_PER_THREAD * compute_iterations_gpu) *
          (long long)compute_grid_size;
  const long long memoryoperations = size;

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
  hipEvent_t start[2], stop[2];

  ComputeSpace cs{size * sizeof(double), compute_iterations_cpu};

  constexpr auto total_bench_iterations = 2;

  float kernel_time_mad_dp_gpu = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start[0], &stop[0]);

    for(int i=0; i<10; i++){
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<double, BLOCK_SIZE, ELEMENTS_PER_THREAD, compute_iterations_gpu>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start[0], stop[0], 0, -2.0f, cd);
    }
    auto kernel_time_mad_dp = benchmark_max_omp([&] {
      return measure_operation([&] {
        bench<double, compute_iterations_cpu>(cs.element_count<double>(), 1., -1., c);
      });
    });
    return 21*kernel_time_mad_dp/finalizeEvents_ext(start[0], stop[0]);
  });
  num_iters=kernel_time_mad_dp_gpu;
}


template <unsigned int compute_iterations_gpu>
void runbench_gpu(double* cd, long size) {
  const long compute_grid_size = size / ELEMENTS_PER_THREAD;
  const int BLOCK_SIZE = 256;
  const int TOTAL_BLOCKS = compute_grid_size / BLOCK_SIZE;
  const long long computations =
      ELEMENTS_PER_THREAD * (long long)compute_grid_size +
      (2 * ELEMENTS_PER_THREAD * compute_iterations_gpu) *
          (long long)compute_grid_size;
  const long long memoryoperations = size;

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
  hipEvent_t start[2], stop[2];

  constexpr auto total_bench_iterations = 6;
  auto kernel_time_mad_dp_gpu = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start[0], &stop[0]);
    
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<double, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations_gpu>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start[0], stop[0], 0, 1.0f, cd);
    return finalizeEvents_ext(start[0], stop[0]);
  });

  printf(
      "GPU,     %4d,   %8.3f, %8.2f, %8.2f, %7.2f\n",
      compute_iterations_gpu,
        // DP
       ((double)computations) / ((double)memoryoperations * sizeof(double)),
       kernel_time_mad_dp_gpu,
       ((double)computations) / kernel_time_mad_dp_gpu * 1000. /
           (double)(1000 * 1000 * 1000),
       ((double)memoryoperations * sizeof(double)) / kernel_time_mad_dp_gpu * 1000. /
          (1000. * 1000. * 1000.)
         );
}

template <unsigned int compute_iterations_cpu, unsigned int compute_iterations_gpu>
void runbench_cpu_gpu_cont(double* cd, double* c, long size) {
  const long compute_grid_size = size / ELEMENTS_PER_THREAD;
  const int BLOCK_SIZE = 256;
  const int TOTAL_BLOCKS = compute_grid_size / BLOCK_SIZE;
  const long long computations =
      ELEMENTS_PER_THREAD * (long long)compute_grid_size +
      (2 * ELEMENTS_PER_THREAD * compute_iterations_gpu) *
          (long long)compute_grid_size;
  const long long memoryoperations = size;

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
  hipEvent_t start[2], stop[2];

  int num_iters=1;
  runbench_iters_gpu<compute_iterations_cpu,compute_iterations_gpu>(cd, c, size, num_iters);
  ComputeSpace cs{size * sizeof(double), compute_iterations_cpu};
  int n_b_iter=1;

  constexpr auto total_bench_iterations = 2;
  auto kernel_time_mad_dp_gpu_dummy = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start[0], &stop[0]);
    
    roctxMark("ROCTX-MARK: before hipLaunchKernel");
    roctxRangePush("ROCTX-RANGE: hipLaunchKernel");

    roctx_range_id_t roctx_id = roctxRangeStartA("roctx_range with id");
    
    for(int i=0; i<2*num_iters; i++){
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<double, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations_gpu>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start[0], stop[0], 0, 1.0f, cd);
    }
    roctxRangeStop(roctx_id);
    roctxMark("ROCTX-MARK: after hipLaunchKernel");
    //hipStreamSynchronize(0); 
    // CPU kernel
    roctxRangePush("ROCTX-RANGE: cpuKernel");

    auto kernel_time_mad_dp_cpu = benchmark_omp([&] {
      return measure_operation([&] {
        bench<double, compute_iterations_cpu>(cs.element_count<double>(), 1., -1., c);
      });
    });
    roctxRangePop();  // for "cpuKernel"
    roctxRangePop();  // for "hipLaunchKernel"
   
    if (n_b_iter==total_bench_iterations){
    const auto computations_dp = cs.compute_ops<double>();
    const auto memory_traffic = cs.memory_traffic();
      printf(
      "CPU,     %4d,   %8.3f, %8.2f, %8.2f, %7.2f\n",
      compute_iterations_cpu,
        // DP
       static_cast<double>(computations_dp) / static_cast<double>(memory_traffic),
       kernel_time_mad_dp_cpu,
       static_cast<double>(computations_dp) / kernel_time_mad_dp_cpu * 1000. /
           static_cast<double>(1000 * 1000 * 1000),
       static_cast<double>(memory_traffic) / kernel_time_mad_dp_cpu * 1000. /
          (1000. * 1000. * 1000.)
         );
    }
    n_b_iter++; 
    return finalizeEvents_ext(start[0], stop[0]);
    //return 0;
  });

  auto kernel_time_mad_dp_gpu = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start[0], &stop[0]);
    
    roctxMark("ROCTX-MARK: before hipLaunchKernel");
    roctxRangePush("ROCTX-RANGE: hipLaunchKernel");

    roctx_range_id_t roctx_id = roctxRangeStartA("roctx_range with id");
    
    for(int i=0; i<num_iters/2; i++){
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<double, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations_gpu>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start[0], stop[0], 0, 1.0f, cd);
    }
    roctxRangeStop(roctx_id);
    roctxMark("ROCTX-MARK: after hipLaunchKernel");
    //hipStreamSynchronize(0); 
    // CPU kernel
    roctxRangePush("ROCTX-RANGE: cpuKernel");

    auto kernel_time_mad_dp_cpu = benchmark_omp([&] {
      return measure_operation([&] {
        bench<double, compute_iterations_cpu>(cs.element_count<double>(), 1., -1., c);
      });
    });
    roctxRangePop();  // for "cpuKernel"
    roctxRangePop();  // for "hipLaunchKernel"
   
    return finalizeEvents_ext(start[0], stop[0]);
  });


  printf(
      "GPU,     %4d,   %8.3f, %8.2f, %8.2f, %7.2f\n",
      compute_iterations_gpu,
        // DP
       ((double)computations) / ((double)memoryoperations * sizeof(double)),
       kernel_time_mad_dp_gpu,
       ((double)computations) / kernel_time_mad_dp_gpu * 1000. /
           (double)(1000 * 1000 * 1000),
       ((double)memoryoperations * sizeof(double)) / kernel_time_mad_dp_gpu * 1000. /
          (1000. * 1000. * 1000.)
         );
  
}



// Variadic template helper to ease multiple configuration invocations
template <unsigned int compute_iterations>
void runbench_range_cpu(double* cd, long size) {
  runbench<compute_iterations>(cd, size);
}

template <unsigned int j1, unsigned int j2, unsigned int... Args>
void runbench_range_cpu(double* cd, long size) {
  runbench_range_cpu<j1>(cd, size);
  runbench_range_cpu<j2, Args...>(cd, size);
}

// Variadic template helper to ease multiple configuration invocations
template <unsigned int compute_iterations>
void runbench_range_gpu(double* cd, long size) {
  runbench_gpu<compute_iterations>(cd, size);
}

template <unsigned int j1, unsigned int j2, unsigned int... Args>
void runbench_range_gpu(double* cd, long size) {
  runbench_range_gpu<j1>(cd, size);
  runbench_range_gpu<j2, Args...>(cd, size);
}

template <unsigned int compute_iterations_gpu>
void runbench_range_cg(double* cd, double* c, long size, unsigned int compute_iterations_cpu) {
  
  runbench_cpu_gpu_cont<CPU_COMPUTE_ITERS_CONTENTION,compute_iterations_gpu>(cd, c, size);
}

template <unsigned int j1, unsigned int j2, unsigned int... Args>
void runbench_range_cg(double* cd, double* c, long size, unsigned int compute_iterations_cpu) {
  runbench_range_cg<j1>(cd, c, size, compute_iterations_cpu);
  runbench_range_cg<j2, Args...>(cd, c, size, compute_iterations_cpu);
}


void mixbenchCPU(double* c, size_t size, int* mod_opt) {
// Initialize data to zeros on memory by respecting 1st touch policy
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++)
    c[i] = 0.0;

  std::cout << "--------------------------------------------"
               "-------------- CSV data "
               "--------------------------------------------"
               "--------------"
            << std::endl;
  std::cout << "Experiment ID, Double Precision ops,,,,              "
            << std::endl;
  std::cout << "CPU or GPU, Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, "
            << std::endl;


  double* cd;

  HIP_SAFE_CALL(hipMalloc((void**)&cd, size * sizeof(double)));

  // Copy data to device memory
  HIP_SAFE_CALL(
      hipMemset(cd, 0, size * sizeof(double)));  // initialize to zeros

  // Synchronize in order to wait for memory operations to finish
  HIP_SAFE_CALL(hipDeviceSynchronize());

  // Copy results to device memory
  //HIP_SAFE_CALL(hipMemcpy(cd, c, size * sizeof(double), hipMemcpyHostToDevice));
  HIP_SAFE_CALL(hipDeviceSynchronize());

  runbench_warmup(cd, size);

  if (mod_opt[0]){
    runbench_range_cpu<0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 6 * 8, 7 * 8,
                  8 * 8, 10 * 8, 13 * 8, 15 * 8, 16 * 8, 20 * 8, 24 * 8, 32 * 8,
                   40 * 8, 64 * 8>(c, size);
  }else if (mod_opt[1]){
    runbench_range_gpu<0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 6 * 8, 7 * 8,
                  8 * 8, 10 * 8, 13 * 8, 15 * 8, 16 * 8, 20 * 8, 24 * 8, 32 * 8,
                   40 * 8, 64 * 8>(cd, size);
  }else if (mod_opt[2]){
    unsigned int cpu_f = 32 * 8;
    runbench_range_cg<0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 6 * 8, 7 * 8,
                  8 * 8, 10 * 8, 13 * 8, 15 * 8, 16 * 8, 20 * 8, 24 * 8, 32 * 8,
                   40 * 8, 64 * 8>(cd, c, size, cpu_f);
  }
	

  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------------"
            << std::endl;
}
