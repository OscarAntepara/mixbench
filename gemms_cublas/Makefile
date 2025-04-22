# BLAS = [ CBLAS | MKL | CUBLAS | HIPBLAS ]
BLAS=CUBLAS
# default is OpenMP
ifeq ($(BLAS),CBLAS)
	CC = g++
	CFLAGS = -std=c++11 -O3 -DUSE_CBLAS
	# modify to suit your platform
	CFLAGS += -I/global/cfs/cdirs/m2956/nanding/software/OpenBLAS/include
	LIBS = -L/global/cfs/cdirs/m2956/nanding/software/OpenBLAS/lib/ -lopenblas
else ifeq ($(BLAS),MKL)
	CC = icpc
	CFLAGS = -std=c++11 -O3 -DUSE_MKL -mkl=parallel -qopenmp
else ifeq ($(BLAS),CUBLAS)
	CC = nvcc
	CFLAGS = --std c++11 -O3 -DUSE_CUBLAS -I$(CUDA_ROOT)/include
	LIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64 -lcublas
else ifeq ($(BLAS),HIPBLAS)
	CC = hipcc
	CFLAGS = -std=c++11 -O3 -DUSE_HIPBLAS
	LIBS = -L$(ROCM_PATH)/hipblas/lib -lhipblas
else
	CC = g++
	CFLAGS = -std=c++11 -O3 -fopenmp
endif

#uncomment if you want to enable MPI
#CC = nvc++ 
#CFLAGS += #-DUSE_MPI

EXE = mt-dgemm.x

# PRECISION = [ 1 | 2 | 3 ]
# 1 is single-precision
# 2 is double-precision
# 3 is half-precision
ifeq ($(PRECISION),1)
	CFLAGS += -DPRECISION=1
	EXE = mt-sgemm.x
endif

ifeq ($(PRECISION),2)
	CFLAGS += -DPRECISION=2
endif

ifeq ($(PRECISION),3)
        CFLAGS += -DPRECISION=3
        EXE = mt-hgemm.x
endif

ifeq ($(RANDOM),1)
        CFLAGS += -DUSE_RANDOM=1
endif

$(EXE): mt-gemm.cpp Makefile
	$(CC) $(CFLAGS) -o $(EXE) mt-gemm.cpp $(LIBS)

clean:
	rm -f *.x

