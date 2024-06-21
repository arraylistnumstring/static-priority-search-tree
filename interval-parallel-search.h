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
	intervalParallelSearchGlobal<<<, ,>>>(pt_arr_d, num_elems, res_pt_arr_d, search_val, false);
	
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
	intervalParallelSearchGlobal<<<, ,>>>(pt_arr_d, num_elems, res_id_arr_d, search_val, true);

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
// Shared memory must be at least of size (1 + number of warps) * sizeof(unsigned long long), where the 1 stores the block-level offset index of res_arr_d starting at which results are stored (is updated in each iteration)
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__global__ void intervalParallelSearchGlobal(PointStructTemplate<T, IDType, num_IDs> pt_arr_d,
												const size_t num_elems, auto *const res_arr_d,
												const T search_val, const bool reportID)
{
	extern __shared__ unsigned long long s[];
	unsigned long long &block_level_offset = *s;
	unsigned long long *warp_level_num_elems_arr = s + 1;

	const unsigned long long num_warps = blockDim.x / warpSize + (blockDim.x % warpSize == 0) ? 0 : 1;

	unsigned long long thread_level_num_elems;	// Calculated with inclusive prefix sum
	unsigned long long thread_level_offset;		// Calculated with exclusive prefix sum	(i.e. preceding element of inclusive prefix sum result)

	unsigned long long warp_level_offset;		// Calculated with exclusive prefix sum

	// Needs a separate flag, as thread_level_num_elems will not be 0 as long as there is at least one preceding thread with an active cell assigned
	bool cell_active;

	// Liu et al. kernel; iterate over all PointStructTemplate<T, IDType, num_IDs> elements in pt_arr_d
	// Use pragma unroll to decrease register occupation, as the number of loops is known at compile time
#pragma unroll
	for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
			i < num_elems; i += gridDim.x * blockDim.x)
	{
		thread_level_offset = 0;	// Default value

		// Evaluate if current metacell is active; if active, set corresponding flags and integers to signal to warp-scan (prefix sum)
		if (pt_arr_d[i].dim1_val <= search_val && search_val <= pt_arr_d[i].dim2_val)
		{
			thread_level_num_elems = 1;
			cell_active = true;
		}
		else
		{
			thread_level_num_elems = 0;
			cell_active = false;
		}

		// Warp-shuffle procedure to calculate inclusive prefix sum, i.e. so current thread knows offset with respect to start index in res_arr_d at which to output result

		// Intra-warp prefix sum
#pragma unroll
		for (unsigned shfl_offset = 1; shfl_offset < warpSize; shfl_offset <<= 1)
		{
			// Copies value of variable thread_level_num_elems from thread with lane ID that is shfl_offset less than current thread's lane ID; no lane ID wrapping occurs, so threads with lane ID lower than shfl_offset will not be affected
			// Threads can only receive data from other threads participating in the __shfl_*sync() call; behavior is undefined when getting data from an inactive thread
			thread_level_num_elems += __shfl_up_sync(0xffffffff, thread_level_num_elems,
														shfl_offset);
		}

		// Exclusive prefix sum result is simply the element in the preceding index of the result of the inclusive prefix sum; note that as each thread is responsible for at most 1 output element, this is effectively thread_level_num_elems - 1_{cell_active}, where 1_{cell_active} is the indicator function for whether the thread's own value of cell_active is true
		// Use of __shfl_up_sync() in this instance is for generality and for ease of calculation
		thread_level_offset = __shfl_up_sync(0xffffffff, thread_level_num_elems, 1);

		// Last thread in warp puts total number of slots necessary for all active cells in the warp_level_num_elems_arr shared memory array
		if constexpr (threadIdx.x % warpSize == warpSize - 1)
			// Place total number of elements in this warp at the slot assigned to this warp in shared memory array warp_level_num_elems_arr
			warp_level_num_elems_arr[threadIdx.x / warpSize] = thread_level_num_elems;

		__syncthreads();	// Warp-level info must be ready to use at the block level
		// Inter-warp prefix sum (block-level)
		// One warp is active in this process; use the first warp, which is guaranteed to exist
		if constexpr (threadIdx.x / warpSize == 0)
		{
			unsigned long long warp_level_num_elems;
			unsigned long long interm_warp_num_elem_offset;
#pragma unroll
			// If necessary, repeat inter-warp prefix sum with base offset of previous iteration until all warps have had their prefix sum calculated
			// j < (size of warp_level_num_elems_arr == number of warps)
			for (unsigned long long j = threadIdx.x; j < num_warps; j += warpSize)
			{
				// Get offset for values in this iteration from the last result of the previous iteration
				interm_warp_num_elem_offset = (j < warpSize) ? 0 :
												warp_level_num_elems_arr[j - threadIdx.x - 1];

				warp_level_num_elems = interm_warp_num_elem_offset + warp_level_num_elems_arr[j];

				// Do inclusive prefix sum on warp-level values
#pragma unroll
				for (unsigned shfl_offset = 1; shfl_offset < warpSize; shfl_offset <<= 1)
				{
					// Copies value of variable warp_level_num_elems from thread with lane ID that is shfl_offset less than current thread's lane ID; no lane ID wrapping occurs, so threads with lane ID lower than shfl_offset will not be affected
					// Threads can only receive data from other threads participating in the __shfl_*sync() call; behavior is undefined when getting data from an inactive thread
					warp_level_num_elems += __shfl_up_sync(0xffffffff, warp_level_num_elems,
																shfl_offset);
				}

				// Store result in shared memory
				warp_level_num_elems_arr[j] = warp_level_num_elems;
			}
		}
		__syncthreads();	// Total number of slots needed to store all active metacells is now known

		// Single thread in block allocates space in res_arr_d with atomic operation
		if constexpr (threadIdx.x == 0)
		{
			const unsigned long long block_level_num_elems = warp_level_num_elems_arr[num_warps - 1];
			block_level_offset = atomicAdd(&res_arr_ind_d, block_level_num_elems);
		}

		// All warps acquire their warp-level offset, which is the element of index (warp ID - 1) in warp_level_num_elems_arr
		// Acquisition of warp-level offset is independent of memory allocation, so can occur anytime after the warp-level __syncthreads() call and before writing results to global memory; placed here for optimisation purposes, as all threads other than thread 0 would otherwise be idle between these __syncthreads() calls
		if constexpr (threadIdx.x / warpSize == 0)
			warp_level_offset = 0;
		else
			warp_level_offset = warp_level_num_elems[threadIdx.x / warpSize - 1];

		__syncthreads();

		// Output to result array
		if (cell_active)
		{
			unsigned long long output_index = block_level_offset + warp_level_offset + thread_level_offset;

			if constexpr (reportID)
			{
				// Add ID to array
				res_arr_d[output_index] = pt_arr_d[i].id;
			}
			else
			{
				// Add PtStructTemplate<T, IDType, num_IDs> to array
				res_arr_d[output_index] = pt_arr_d[i];
			}
		}
	}
};

#endif
