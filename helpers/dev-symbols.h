#ifndef DEV_SYMBOLS_H
#define DEV_SYMBOLS_H

// To use a global memory-scoped variable, must declare it outside of any function
// To match a valid atomicAdd function signature, res_arr_ind_d must be declared as an unsigned long long (unsigned long long is the same as an unsigned long long int)
__device__ unsigned long long res_arr_ind_d;

/*
// No longer useful with the removal of cudaStreamFireAndForget usage during StaticPSTGPU tree construction

// To track number of active resident grids for purpose of determining whether to call dynamic parallel functions with cudaStreamFireAndForget or to simply use the default device stream (as an excessive number of the former causes resource allocation failures and therefore correctness issues)
__device__ unsigned int num_active_grids_d;

// Source for compute capability-specific technical specifications:
//	Table 21 of
//		https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

// __CUDA_ARCH__ is a preprocessor variable defined only on device that evaluates to XY0 for a device of compute capability X.Y
// To save memory (especially duplication of value across host and device), simply declare MAX_NUM_ACTIVE_GRIDS a preprocessor variable
#if __CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 700 || __CUDA_ARCH__ >= 750
	#define MAX_NUM_ACTIVE_GRIDS 128
#elif __CUDA_ARCH__ <= 520 || __CUDA_ARCH__ == 610
	#define MAX_NUM_ACTIVE_GRIDS 32
// Put the lowest number of maximum active grids allowed as the default for architectures not listed here (which follow the list found in the CUDA C++ Programming Guide
#else // __CUDA_ARCH__ == 530 || __CUDA_ARCH__ == 620 || __CUDA_ARCH__ = 720
	#define MAX_NUM_ACTIVE_GRIDS 16
#endif
*/

// Avoid compiler-issued signed int underflow warnings by attaching a "u" postfix to indicate unsigned values
#define MAX_X_DIM_NUM_BLOCKS ((1u << 31) - 1)
#define MAX_Y_DIM_NUM_BLOCKS ((1u << 16) - 1)
#define MAX_Z_DIM_NUM_BLOCKS ((1u << 16) - 1)

#define MAX_X_DIM_THREADS_PER_BLOCK 1024
#define MAX_Y_DIM_THREADS_PER_BLOCK 1024
#define MAX_Z_DIM_THREADS_PER_BLOCK 64

#define MAX_THREADS_PER_BLOCK 1024

#endif
