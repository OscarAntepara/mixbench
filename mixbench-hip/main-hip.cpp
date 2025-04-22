/**
 * main-hip.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lhiputil.h"
#include "mix_kernels_hip.h"
#include "version_info.h"
#include <iostream>

#define VECTOR_SIZE (64 * 1024 * 1024)

int main(int argc, char* argv[]) {
  printf("mixbench-hip (%s)\n", VERSION_INFO);
  int random=0;
  int data_type=0;
  int run_long=0;
  if (argc!=4 || std::atoi(argv[1])>1 || std::atoi(argv[1])<0 || std::atoi(argv[2])>3 || std::atoi(argv[2])<0 || std::atoi(argv[3])>1 || std::atoi(argv[3])<0) {
    std::cout << "Test requires three args. "<< std::endl;
    std::cout << "First arg (0/1) ==> zero/random data "<< std::endl;
    std::cout << "Second arg (0/1/2/3) ==> FP64/FP32/FP16/INT32 "<< std::endl;
    std::cout << "Third arg (0/1) ==> orig/running longer for power measurement "<< std::endl;
    exit (0);
  }
  for (int i = 1; i < argc; ++i) {
    if (i==1 && std::atoi(argv[i])==1) random=1;
    if (i==2 && std::atoi(argv[i])==1) data_type=1;
    if (i==2 && std::atoi(argv[i])==2) data_type=2;
    if (i==2 && std::atoi(argv[i])==3) data_type=3;
    if (i==3 && std::atoi(argv[i])==1) run_long=1;
  }
  std::cout << "Test with ";
  if (random) std::cout <<"random data. ";
  else std::cout <<"zero data. ";
  std::cout <<std::endl;
  std::cout << "Data type ";
  if (data_type==0) std::cout <<"FP64. ";
  else if (data_type==1) std::cout <<"FP32. ";
  else if (data_type==2) std::cout <<"FP16. ";
  else if (data_type==3) std::cout <<"INT32. ";
  std::cout <<std::endl;
  if (run_long) std::cout <<"Power profile mode."<<std::endl;

  unsigned int datasize = VECTOR_SIZE * sizeof(double);

  HIP_SAFE_CALL(hipSetDevice(0));
  StoreDeviceInfo(stdout);

  size_t freeCUDAMem, totalCUDAMem;
  HIP_SAFE_CALL(hipMemGetInfo(&freeCUDAMem, &totalCUDAMem));
  printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
  printf("Buffer size:          %dMB\n", datasize / (1024 * 1024));

  double* c;
  c = (double*)malloc(datasize);
  for (int i = 0; i < VECTOR_SIZE ; i++) {
    if (random) c[i] = 1.0 + ( (double)(rand()) / (double)(RAND_MAX) );
    else c[i] = 0.0;
  }

  mixbenchGPU(c, VECTOR_SIZE, data_type, run_long);

  free(c);

  return 0;
}
