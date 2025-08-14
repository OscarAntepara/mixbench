# mixbench-cpu-gpu-contention with CUDA

This is the initial implementaion to execute CPU and GPU task concurrently.

## Building notes

Load the most recent nvhpc compilers and build following the next line:
nvc++ -Munroll -Mvect -mcpu=neoverse-v2 -fopenmp -v -o exe main.cpp mix_kernels_cpu.cu

Execute with:
OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores ./exe -c

Arguments:
-c Executes the CPU roofline.
-g Executes the GPU roofline.
-cg Executes the GPU roofline with the CPU executing a compute bound or memory bound kernel
-gc Executes the CPU roofline with the GPU executing a compute bound or memory bound kernel
