#ifndef INTERVAL_PARALLEL_SEARCH_H
#define INTERVAL_PARALLEL_SEARCH_H

#include <cooperative_groups.h>
#include <type_traits>

#include "calc-alloc-report-ind-offset.h"
#include "dev-symbols.h"					// For global memory-scoped variable res_arr_ind_d
#include "gpu-err-chk.h"


#ifdef DEBUG_SEARCH
#include <iostream>
#endif


extern __device__ unsigned long long res_arr_ind_d;		// Declared in dev-symbols.h


// Method of Liu et al. (2016): embarrassingly parallel search for active metacells, superficially modified for parity with PST search method

// Given an array of PointStructTemplate<T, IDType, num_IDs>, return an on-device array of either points or IDs, each point pt or ID for point pt satisfies search_val \in [pt.dim1_val, pt.dim2_val])
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
	>::value
void intervalParallelSearch(PointStructTemplate<T, IDType, num_IDs>* pt_arr_d, const size_t num_elems,
							RetType *&res_arr_d, size_t &num_res_elems, T search_val,
							const int dev_ind, const int num_devs, const int warp_size,
							const unsigned num_thread_blocks, const unsigned threads_per_block)
{
	// Allocate space on GPU for output array, whether metacell tags or IDs
	gpuErrorCheck(cudaMalloc(&res_arr_d, num_elems * sizeof(RetType)),
					"Error in allocating array to store PointStruct search result on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	// Set on-device global result array index to 0
	unsigned long long res_arr_ind = 0;
	// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
	gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind, sizeof(size_t)),
					"Error in initialising global result array index to 0 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	const unsigned warps_per_block = threads_per_block / warp_size + ((threads_per_block % warp_size == 0) ? 0 : 1);

#ifdef DEBUG_SEARCH
	std::cout << "Beginning on-device search\n";
#endif

	// Call global function for on-device search
	intervalParallelSearchGlobal<<<num_thread_blocks, threads_per_block, (warps_per_block + 1) * sizeof(unsigned long long)>>>(pt_arr_d, num_elems, res_arr_d, search_val);
	
	// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
	// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
	gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d, sizeof(unsigned long long)),
					"Error in copying global result array final index from device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
};

// Search in parallel for all points pt satisfying search_val \in [pt.dim1_val, pt.dim2_val]; output array is of type IDType if decltype(res_arr_d) = IDType and of type PointStructTemplate<T, IDType, num_IDs> if decltype(res_arr_d) = PointStructTemplate<T, IDType, num_IDs>
// Shared memory must be at least of size (1 + number of warps) * sizeof(unsigned long long), where the 1 stores the block-level offset index of res_arr_d starting at which results are stored (is updated in each iteration)
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
	>::value
__global__ void intervalParallelSearchGlobal(PointStructTemplate<T, IDType, num_IDs> *pt_arr_d,
												const size_t num_elems, RetType *const res_arr_d,
												const T search_val)
{
	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	unsigned long long &block_level_res_start_ind = *reinterpret_cast<unsigned long long *>(s);
	unsigned long long *warp_level_num_elems_arr = reinterpret_cast<unsigned long long *>(s) + 1;

	const cooperative_groups::thread_block curr_block = cooperative_groups::this_thread_block();

	// Liu et al. kernel; iterate over all PointStructTemplate<T, IDType, num_IDs> elements in pt_arr_d
	// Due to passing of curr_block to calcAllocReportIndOffset(), whole block must iterate if at least one thread has an element to process
	// Loop unrolling, as number of loops is known explicitly when kernel is called
#pragma unroll
	for (unsigned long long i = linBlockID() * curr_block.num_threads();
			i < num_elems; i += gridDim.x * gridDim.y * gridDim.z * curr_block.num_threads())
	{
		const bool active_cell = i + curr_block.thread_rank() < num_elems
							&& pt_arr_d[i + curr_block.thread_rank()].dim1_val <= search_val
							&& search_val <= pt_arr_d[i + curr_block.thread_rank()].dim2_val;


		const unsigned long long thread_offset_in_block
				= calcAllocReportIndOffset<unsigned long long>(curr_block, active_cell ? 1 : 0,
																warp_level_num_elems_arr,
																block_level_res_start_ind);

		// Output to result array
		if (active_cell)
		{
			if constexpr (std::is_same<RetType, IDType>::value)
				// Add ID to array
				res_arr_d[block_level_res_start_ind + thread_offset_in_block]
							= pt_arr_d[i + curr_block.thread_rank()].id;
			else
				// Add PtStructTemplate<T, IDType, num_IDs> to array
				res_arr_d[block_level_res_start_ind + thread_offset_in_block]
							= pt_arr_d[i + curr_block.thread_rank()];
		}
	}
};

#endif
