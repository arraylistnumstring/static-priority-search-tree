#ifndef INTERVAL_PARALLEL_SEARCH_H
#define INTERVAL_PARALLEL_SEARCH_H

#include "dev-symbols.h"	// For global memory-scoped variable res_arr_ind_d
#include "gpu-err-chk.h"

// Method of Liu et al. (2016): embarrassingly parallel search for active metacells, superficially modified for parity with PST search method

// Given an array of PointStructTemplate<T, IDType, num_IDs>, return an on-device array of PointStructTemplate<T, IDType, num_IDs> where each point pt satisfies search_val \in [pt.dim1_val, pt.dim2_val])
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
PointStructTemplate<T, IDType, num_IDs>* intervalParallelSearch(PointStructTemplate<T, IDType, num_IDs>* pt_arr, const size_t num_pts, size_t &num_res_elems, T search_val, const int dev_ind, const int num_devs)
{
	// Allocate space on GPU for input metacell tag array and copy to device
	PointStructTemplate<T, IDType, num_IDs>* pt_arr_d;
	gpuErrorCheck(cudaMalloc(&pt_arr_d, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>)),
					"Error in allocating array to store initial PointStructs on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");
	gpuErrorCheck(cudaMemcpy(pt_arr_d, pt_arr, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>)),
					"Error in copying array of PointStructTemplate<T, IDType, num_IDs> objects to device "
                                                + std::to_string(dev_ind) + " of " + std::to_string(num_devs)
                                                + ": ");

	// Allocate space on GPU for output metacell tag array
	PointStructTemplate<T, IDType, num_IDs>* res_pt_arr_d;
	gpuErrorCheck(cudaMalloc(&res_pt_arr_d, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>)),
					"Error in allocating array to store PointStruct search result on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	// Set on-device global result array index to 0
	unsigned long long res_arr_ind = 0;
	// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
	gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind, sizeof(size_t),
										0, cudaMemcpyDefault),
					"Error in initialising global result array index to 0 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	// Call global function for on-device search
	intervalParallelSearchGlobal<<<>>>();
	
	// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
	// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
	gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d,
										sizeof(unsigned long long), 0, cudaMemcpyDefault),
					"Error in copying global result array final index from device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	// Return device pointer in case more on-device computations need to be done, e.g. Marching Cubes
	return res_pt_arr_d;
};

// Given an array of PointStructTemplate<T, IDType, num_IDs>, return an on-device array of indices where each index i satisfies search_val \in [pt_arr[i].dim1_val, pt_arr[i].dim2_val])
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs=1>
IDType* intervalParallelSearchID(PointStructTemplate<T, IDType, num_IDs>* pt_arr, const size_t num_pts, size_t &num_res_elems, T search_val, const int dev_ind, const int num_devs)
{
	// Allocate space on GPU for input metacell tag array and copy to device
	PointStructTemplate<T, IDType, num_IDs>* pt_arr_d;
	gpuErrorCheck(cudaMalloc(&pt_arr_d, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>)),
					"Error in allocating array to store initial PointStructs on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");
	gpuErrorCheck(cudaMemcpy(pt_arr_d, pt_arr, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>)),
					"Error in copying array of PointStructTemplate<T, IDType, num_IDs> objects to device "
                                                + std::to_string(dev_ind) + " of " + std::to_string(num_devs)
                                                + ": ");

	// Allocate space on GPU for output metacell IDs
	IDType* res_id_arr_d;
	gpuErrorCheck(cudaMalloc(&res_id_arr_d, num_elems * sizeof(size_t),
					"Error in allocating array to store PointStruct ID search result on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	// Set on-device global result array index to 0
	unsigned long long res_arr_ind = 0;
	// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
	gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind, sizeof(size_t),
										0, cudaMemcpyDefault),
					"Error in initialising global result array index to 0 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	// Call global function for on-device search
	intervalParallelSearchGlobal<<<>>>();

	// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
	// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
	gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d,
										sizeof(unsigned long long), 0, cudaMemcpyDefault),
					"Error in copying global result array final index from device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	// Return device pointer in case more on-device computations need to be done, e.g. Marching Cubes
	return res_id_arr_d;
};

// Search in parallel for all points pt satisfying search_val \in [pt.dim1_val, pt.dim2_val]; output array is of type IDType if reportID = true and of type PointStructTemplate<T, IDType, num_IDs> if reportID = false
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__global__ void intervalParallelSearchGlobal(PointStructTemplate<T, IDType, num_IDs> pt_arr_d,
												const size_t num_elems, auto *const res_arr_d,
												const bool reportID)
{
	// Liu et al. kernel; iterate over all PointStructTemplate<T, IDType, num_IDs>
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
			i < num_elems; i += gridDim.x * blockDim.x)
	if constexpr (reportID)
	{
		// Add ID to array
	}
	else
	{
		// Add PtStructTemplate<T, IDType, num_IDs> to 
	}
};

#endif
