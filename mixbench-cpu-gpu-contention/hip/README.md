# mixbench-cpu-gpu-contention with HIP

This is the initial implementaion to execute CPU and GPU task concurrently.

## Building notes

Load the most recent rocm or AMD compilers and build following the next line:
hipcc -march=native -funroll-loops -fopenmp -v -o exe main.cpp mix_kernels_cpu.cpp

Execute with:
OMP_NUM_THREADS=48 OMP_PROC_BIND=true ./exe -c

Arguments:
-c Executes the CPU roofline.
-g Executes the GPU roofline.
-cg Executes the GPU roofline with the CPU executing a compute bound or memory bound kernel
-gc Executes the CPU roofline with the GPU executing a compute bound or memory bound kernel
