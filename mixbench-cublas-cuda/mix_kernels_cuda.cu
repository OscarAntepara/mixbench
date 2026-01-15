/**
 * mix_kernels_cuda_ro.cu: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <typeinfo> // For type identification
#include "lcutil.h"

#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define ELEMENTS_PER_THREAD (8)
#define FUSION_DEGREE (4)

template<class T>
inline __device__ T conv_int(const int i){ return static_cast<T>(i); }

template<class T>
inline __device__ T mad(const T a, const T b, const T c){ return a*b+c; }

template<class T>
inline __device__ bool equal(const T a, const T b){ return a==b; }

#if __CUDA_ARCH__ >= 530
template<>
inline __device__ half2 conv_int(const int i){ return __half2half2( __int2half_rd(i) ); }
template<>
inline __device__ half2 mad(const half2 a, const half2 b, const half2 c){ return __hfma2(a, b, c)/*__hadd2(__hmul2(a, b), c)*/; }
template<>
inline __device__ bool equal(const half2 a, const half2 b){ return __hbeq2(a, b); }
#else
// a dummy implementations as a workaround
template<>
inline __device__ half2 conv_int(const int i){ return half2(); }
template<>
inline __device__ half2 mad(const half2 a, const half2 b, const half2 c){ return half2(); }
template<>
inline __device__ bool equal(const half2 a, const half2 b){ return false; }
#endif

template <class T, int blockdim, unsigned int granularity, unsigned int fusion_degree, unsigned int compute_iterations, bool TemperateUnroll>
__global__ void benchmark_func(T seed, T *g_data){
	const unsigned int blockSize = blockdim;
	const int stride = blockSize;
	int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
	const int big_stride = gridDim.x*blockSize*granularity;

	T tmps[granularity];
	for(int k=0; k<fusion_degree; k++){
		#pragma unroll
		for(int j=0; j<granularity; j++){
			// Load elements (memory intensive part)
			tmps[j] = g_data[idx+j*stride+k*big_stride];
			// Perform computations (compute intensive part)
			#pragma unroll TemperateUnroll ? 4 : 128
			for(int i=0; i<compute_iterations; i++){
				tmps[j] = mad(tmps[j], tmps[j], seed);
			}
		}
		// Multiply add reduction
		T sum = conv_int<T>(0);
		#pragma unroll
		for(int j=0; j<granularity; j+=2)
			sum = mad(tmps[j], tmps[j+1], sum);
		// Dummy code
		if( equal(sum, conv_int<T>(-1)) ) // Designed so it never executes
			g_data[idx+k*big_stride] = sum;
	}
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

void runbench_warmup(double *cd, long size){
	const long reduced_grid_size = size/(ELEMENTS_PER_THREAD)/128;
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< short, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, 0, true ><<< dimReducedGrid, dimBlock >>>((short)1, (short*)cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

// ------------------------------------------------------- //
// Function: get_seconds
// ------------------------------------------------------- //
double get_seconds() {

  struct timeval now;
  gettimeofday(&now, NULL);

  const double seconds = (double) now.tv_sec;
  const double usec    = (double) now.tv_usec;

  return seconds + (usec * 1.0e-6);
}

template <typename gemm_t>
static inline double calc_gemm(int repeats, int N, gemm_t alpha, gemm_t beta, 
                             gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC) {
  
  cudaError_t errorA, errorB, errorC;
  gemm_t *d_matrixA, *d_matrixB, *d_matrixC;
  errorA = cudaMalloc ((void**)&d_matrixA, N*N*sizeof(gemm_t));
  errorB = cudaMalloc ((void**)&d_matrixB, N*N*sizeof(gemm_t));
  errorC = cudaMalloc ((void**)&d_matrixC, N*N*sizeof(gemm_t));
  if(  (errorA != cudaSuccess)
    || (errorB != cudaSuccess) 
    || (errorC != cudaSuccess) ) { 
    printf("ERROR: allocating device matrices\n");
    exit(1);
  }
  
  cublasStatus_t status;
  cublasHandle_t handle;
  status = cublasCreate(&handle);
  if( status != CUBLAS_STATUS_SUCCESS ) { 
    printf("ERROR: creating a device handle\n"); 
    exit(1); 
  }

  //status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
  if constexpr (std::is_same_v<gemm_t, float>) status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  cublasStatus_t statusA, statusB, statusC;
  statusA = cublasSetMatrix (N, N, sizeof(gemm_t), matrixA, N, d_matrixA, N);
  statusB = cublasSetMatrix (N, N, sizeof(gemm_t), matrixB, N, d_matrixB, N);
  statusC = cublasSetMatrix (N, N, sizeof(gemm_t), matrixC, N, d_matrixC, N);
  if(  (statusA != CUBLAS_STATUS_SUCCESS)
    || (statusB != CUBLAS_STATUS_SUCCESS)
    || (statusC != CUBLAS_STATUS_SUCCESS) ) {
    printf("ERROR: intializing device matrices\n");
    exit(1);
  }

  // Repeat multiple times
  const double start = get_seconds();
  for (int r = 0; r < repeats; r++) {
    if constexpr (std::is_same_v<gemm_t, float>) {
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                 &alpha, d_matrixA, N, d_matrixB, N, &beta, d_matrixC, N );
    }else if constexpr (std::is_same_v<gemm_t, half>){
    cublasHgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                 &alpha, d_matrixA, N, d_matrixB, N, &beta, d_matrixC, N );
    }else{
    cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                 &alpha, d_matrixA, N, d_matrixB, N, &beta, d_matrixC, N );
    }
  }
  cudaDeviceSynchronize();
  const double end = get_seconds();

  cublasGetMatrix(N, N, sizeof(gemm_t), d_matrixC, N, matrixC, N);
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);
  cublasDestroy(handle);

  return(end-start);
}

template <typename gemm_t>
void run_gemm() {
  int N = 8192;
  int repeats = 200;
  if constexpr (std::is_same_v<gemm_t, float>) repeats=repeats*4;
  if constexpr (std::is_same_v<gemm_t, half>) repeats=repeats*16;
  
  gemm_t alpha = 1.0;
  gemm_t beta  = 0.0;

  printf("Allocating Matrices...\n");

  gemm_t* __restrict__ matrixA = (gemm_t*) malloc(sizeof(gemm_t) * N * N);
  gemm_t* __restrict__ matrixB = (gemm_t*) malloc(sizeof(gemm_t) * N * N);
  gemm_t* __restrict__ matrixC = (gemm_t*) malloc(sizeof(gemm_t) * N * N);

  printf("Allocation complete, populating with values...\n");

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      //matrixA[i*N + j] = 2.0;
      //matrixB[i*N + j] = 0.5;
      //matrixC[i*N + j] = 1.0;
      matrixA[i*N + j] = 0.0;
      matrixB[i*N + j] = 0.0;
      matrixC[i*N + j] = 0.0;
      //matrixA[i*N + j] = rand(); //2.0;
      //matrixB[i*N + j] = rand(); //0.5;
      //matrixC[i*N + j] = rand(); //1.0;   
      matrixA[i*N + j] = 1.0 + (1.0*( (double)(rand()) / (double)(RAND_MAX) ));
      matrixB[i*N + j] = 1.0 + (1.0*( (double)(rand()) / (double)(RAND_MAX) ));
      matrixC[i*N + j] = 1.0 + (1.0*( (double)(rand()) / (double)(RAND_MAX) ));
    }
  }

  printf("Performing TC ");
  if constexpr (std::is_same_v<gemm_t, double>) printf("FP64 ");
  if constexpr (std::is_same_v<gemm_t, float>) printf("FP32 ");
  if constexpr (std::is_same_v<gemm_t, half>) printf("FP16 ");
  printf("multiplication...\n");

  const double time_taken = calc_gemm<gemm_t>(repeats, N, alpha, beta, matrixA, matrixB, matrixC);

  // Print results
  printf("\n");
  printf("===============================================================\n");

  double N_dbl = (double) N;
  double matrix_memory = (3 * N_dbl * N_dbl) * ((double) sizeof(gemm_t));
  printf("Memory for Matrices:  %f MB\n", (matrix_memory / (1024 * 1024)));

  printf("Multiply time:        %f seconds\n", time_taken);

  int mpi_size = 1;
  const double flops_computed = ( (N_dbl * N_dbl * N_dbl * 2.0 * (double)(repeats)) +
                                  (N_dbl * N_dbl * 3 * (double)(repeats)) ) * (double)(mpi_size);

  printf("GFLOP/s rate:         %f GF/s\n", (flops_computed / time_taken) / 1.0e9);

  printf("===============================================================\n");
  printf("\n");

  free(matrixA);
  free(matrixB);
  free(matrixC);





}	

int out_config = 1;

template<unsigned int compute_iterations>
void runbench(double *cd, long size, bool doHalfs, int data_type, int run_long){
	const long compute_grid_size = size/ELEMENTS_PER_THREAD/FUSION_DEGREE;
	//const long compute_grid_size = 8192*256;
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = compute_grid_size/BLOCK_SIZE;
	const long long computations = (ELEMENTS_PER_THREAD*(long long)compute_grid_size+(2*ELEMENTS_PER_THREAD*compute_iterations)*(long long)compute_grid_size)*FUSION_DEGREE;
	const long long memoryoperations = size;

        //printf("total blocks  %4d,  block size  %4d \n", TOTAL_BLOCKS, BLOCK_SIZE);
	int num_iter=200;
	if (compute_iterations>70) num_iter=100;
	if (compute_iterations>190) num_iter=50;
	if (!run_long) num_iter=1;
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
	cudaEvent_t start, stop;

	if (data_type==1){
	  initializeEvents(&start, &stop);
	  //benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(-2.0f, (float*)cd);
	  for(int k=0; k<num_iter; k++) benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(-2.0f, (float*)cd);
	  float kernel_time_mad_sp = finalizeEvents(start, stop)/num_iter;
	  //for(int k=0; k<num_iter; k++) benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(-2.0f, (float*)cd);
          printf("         %4d,   %8.3f,%8.5f,%8.2f,%7.2f\n",
                compute_iterations,
               ((double)computations)/((double)memoryoperations*sizeof(float)),
               kernel_time_mad_sp,
               ((double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
               ((double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.)
                 );
	  
	}

	if (data_type==0){
	  float kernel_time_mad_dp = 0;
	  //for(int k=0; k<num_iter; k++) {
	  initializeEvents(&start, &stop);
	  
	  for(int k=0; k<num_iter; k++) benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(-2.0, cd);

	  //benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(-2.0, cd);
	   cudaDeviceSynchronize();
	  kernel_time_mad_dp = finalizeEvents(start, stop)/num_iter;
	  //}
	  //for(int k=0; k<num_iter; k++) benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(-2.0, cd);
          printf("         %4d,   %8.3f,%8.5f,%8.2f,%7.2f\n",
                compute_iterations,
                ((double)computations)/((double)memoryoperations*sizeof(double)),
                kernel_time_mad_dp,
                ((double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
                ((double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.)
                );

	}

	if (data_type==2){
	  float kernel_time_mad_hp = 0.f;
	  if( doHalfs ){
		initializeEvents(&start, &stop);
		half2 h_ones;
		half2 h_minus2;
		const float dec = -2.0f;
		h_minus2 =  __float2half2_rn(dec);
		*((int32_t*)&h_ones) = 15360 + (15360 << 16); // 1.0 as half
		benchmark_func< half2, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(h_minus2, (half2*)cd);
		kernel_time_mad_hp = finalizeEvents(start, stop);
		for(int k=0; k<num_iter; k++) benchmark_func< half2, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(h_minus2, (half2*)cd);
	  }
          printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f\n",
                compute_iterations,
                ((double)computations)/((double)memoryoperations*sizeof(half2)),
                kernel_time_mad_hp,
                ((double)computations)/kernel_time_mad_hp*1000./(double)(1000*1000*1000),
                ((double)memoryoperations*sizeof(half2))/kernel_time_mad_hp*1000./(1000.*1000.*1000.)
                 );
	  
	}

	if (data_type==3){
 	  initializeEvents(&start, &stop);
	  for(int k=0; k<num_iter; k++)  benchmark_func< int, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, true ><<< dimGrid, dimBlock >>>(-2.0, (int*)cd);
	  float kernel_time_mad_int = finalizeEvents(start, stop)/num_iter;
          printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f\n",
                compute_iterations,
                ((double)computations)/((double)memoryoperations*sizeof(int)),
                kernel_time_mad_int,
                ((double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
                ((double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.)          
                 );
	  
        }
}

extern "C" void mixbenchGPU(double *c, long size, int data_type, int run_long){
	const char *benchtype = "compute with global memory (block strided)";

	printf("Trade-off type:       %s\n", benchtype);
	printf("Elements per thread:  %d\n", ELEMENTS_PER_THREAD);
	printf("Thread fusion degree: %d\n", FUSION_DEGREE);
	bool doHalfs = IsFP16Supported();
	if( !doHalfs )
		printf("Warning:              Half precision computations are not supported\n");

	double *cd;
	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(double)) );

	// Copy data to device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(double)) );  // initialize to zeros

	CUDA_SAFE_CALL( cudaMemcpy(cd, c, size*sizeof(double), cudaMemcpyHostToDevice) );
	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	printf("----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------\n");
	printf("Experiment ID, ops,,,, \n");
	printf("Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec\n");

	runbench_warmup(cd, size);

	runbench<0>(cd, size, doHalfs, data_type, run_long);
	runbench<1>(cd, size, doHalfs, data_type, run_long);
	runbench<2>(cd, size, doHalfs, data_type, run_long);
	runbench<3>(cd, size, doHalfs, data_type, run_long);
	runbench<4>(cd, size, doHalfs, data_type, run_long);
	runbench<5>(cd, size, doHalfs, data_type, run_long);
	runbench<6>(cd, size, doHalfs, data_type, run_long);
	runbench<7>(cd, size, doHalfs, data_type, run_long);
	runbench<8>(cd, size, doHalfs, data_type, run_long);
	runbench<9>(cd, size, doHalfs, data_type, run_long);
	runbench<10>(cd, size, doHalfs, data_type, run_long);
	runbench<11>(cd, size, doHalfs, data_type, run_long);
	runbench<12>(cd, size, doHalfs, data_type, run_long);
	runbench<13>(cd, size, doHalfs, data_type, run_long);
	runbench<14>(cd, size, doHalfs, data_type, run_long);
	runbench<15>(cd, size, doHalfs, data_type, run_long);
	runbench<16>(cd, size, doHalfs, data_type, run_long);
	runbench<17>(cd, size, doHalfs, data_type, run_long);
	runbench<18>(cd, size, doHalfs, data_type, run_long);
	runbench<20>(cd, size, doHalfs, data_type, run_long);
	runbench<22>(cd, size, doHalfs, data_type, run_long);
	runbench<24>(cd, size, doHalfs, data_type, run_long);
	runbench<28>(cd, size, doHalfs, data_type, run_long);
	runbench<32>(cd, size, doHalfs, data_type, run_long);
	runbench<40>(cd, size, doHalfs, data_type, run_long);
	runbench<48>(cd, size, doHalfs, data_type, run_long);
	runbench<56>(cd, size, doHalfs, data_type, run_long);
	runbench<64>(cd, size, doHalfs, data_type, run_long);
	runbench<80>(cd, size, doHalfs, data_type, run_long);
	runbench<96>(cd, size, doHalfs, data_type, run_long);
	runbench<128>(cd, size, doHalfs, data_type, run_long);
	runbench<192>(cd, size, doHalfs, data_type, run_long);
	runbench<256>(cd, size, doHalfs, data_type, run_long);
	runbench<512>(cd, size, doHalfs, data_type, run_long);
	runbench<1024>(cd, size, doHalfs, data_type, run_long);

	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
/*
  run_gemm<double>();
  run_gemm<float>();
  run_gemm<half>();
*/
	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree(cd) );

	CUDA_SAFE_CALL( cudaDeviceReset() );
}
