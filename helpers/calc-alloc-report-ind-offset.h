#ifndef CALC_ALLOC_REPORT_IND_OFFSET_H
#define CALC_ALLOC_REPORT_IND_OFFSET_H

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
// Allows use of nvstd::function, an equivalent to std::function that functions on both host and device (but not across the host-device boundary)
#include <nvfunctional>

#include "dev-symbols.h"	// Need access to __device__ (global memory) variable res_arr_ind_d
#include "linearise-id.h"


extern __device__ unsigned long long res_arr_ind_d;		// Declared in dev-symbols.h


// Intrawarp version: allocates space in output array and returns for each thread the index res_arr_output_ind at which it can safely write its result to the output array
template <typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T calcAllocReportIndOffset(const T curr_thread_num_elems)
{
	// Intrawarp cooperative group
	const cooperative_groups::coalesced_group intrawarp_active_threads
			= cooperative_groups::coalesced_threads();

	// Exclusive prefix sum result is simply the element in the preceding index of the result of the inclusive prefix sum; note that as each thread is responsible for at most 1 output element, this is effectively curr_thread_num_elems - 1_{active_data}, where 1_{active_data} is the indicator function for whether the thread's own value of active_data is true
	// Because exclusive_scan() takes as its second argument a trivially copyable type, the type is also trivially moveable, i.e. its trivial move constructor does the same ting as the trivial copy constructor (which is to simply copy the underlying bytes from one location to another, non-overlapping location), which thereby means the original value is left undisturbed
	const T thread_offset_in_warp = cooperative_groups::exclusive_scan(intrawarp_active_threads, curr_thread_num_elems);

	// Intrawarp: allocate space in output array on per-warp basis
	T res_arr_output_ind;

	// Last thread in coalesced group allocates total number of slots necessary for all active cells in the output array
	const auto last_thread_in_group = intrawarp_active_threads.num_threads() - 1;
	if (intrawarp_active_threads.thread_rank() == last_thread_in_group)
		// Warp-level start index
		res_arr_output_ind = atomicAdd(&res_arr_ind_d, thread_offset_in_warp + curr_thread_num_elems);

	// Broadcast warp-level start index to other threads in warp (i.e. all active threads read from last active thread in the warp)
	res_arr_output_ind = intrawarp_active_threads.shfl(res_arr_output_ind, last_thread_in_group);

	// Add thread-level offset to warp-level start index to find output index for this thread
	return res_arr_output_ind + thread_offset_in_warp;
}

// Intrawarp + interwarp version: allocates space in output arra; stores first index allocated for the block in warp_level_num_elems_arr
template <typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T calcAllocReportIndOffset(const cooperative_groups::thread_block &curr_block,
														const T curr_thread_num_elems,
														T *const warp_level_num_elems_arr,
														T &block_level_res_start_ind
													)
{
	// Intrawarp cooperative group
	const cooperative_groups::coalesced_group intrawarp_active_threads
			= cooperative_groups::coalesced_threads();

	// Exclusive prefix sum result is simply the element in the preceding index of the result of the inclusive prefix sum; note that as each thread is responsible for at most 1 output element, this is effectively curr_thread_num_elems - 1_{active_data}, where 1_{active_data} is the indicator function for whether the thread's own value of active_data is true
	// Because exclusive_scan() takes as its second argument a trivially copyable type, the type is also trivially moveable, i.e. its trivial move constructor does the same ting as the trivial copy constructor (which is to simply copy the underlying bytes from one location to another, non-overlapping location), which thereby means the original value is left undisturbed
	const T thread_offset_in_warp = cooperative_groups::exclusive_scan(intrawarp_active_threads, curr_thread_num_elems);

	// Last active thread in warp puts total number of slots necessary for all active cells in the warp_level_num_elems_arr shared memory array
	if (intrawarp_active_threads.thread_rank() == intrawarp_active_threads.num_threads() - 1)
		// Place total number of elements in this warp at the slot assigned to this warp in shared memory array warp_level_num_elems_arr
		warp_level_num_elems_arr[curr_block.thread_rank() / warpSize] = thread_offset_in_warp + curr_thread_num_elems;

	curr_block.sync();		// Warp-level info must be ready to use at the block level


	// Interwarp prefix sum (block-level)
	const auto warps_per_block = curr_block.num_threads() / warpSize
									+ (curr_block.num_threads() % warpSize == 0 ? 0 : 1);

	// Being a prefix sum, only one warp should be active in this process for speed and correctness; use the first warp, which is guaranteed to exist
	if (curr_block.thread_rank() / warpSize == 0)
	{
		// If necessary, repeat interwarp prefix sum with base offset of previous iteration until all warps have had their prefix sum calculated
#pragma unroll
		for (auto i = 0; i < warps_per_block; i += warpSize)
		{
			// Inter-warp condition
			if (i + curr_block.thread_rank() < warps_per_block)
			{
				// Interwarp cooperative group; must used coalesced_groups in order to use inclusive_scan()
				const cooperative_groups::coalesced_group interwarp_active_threads
						= cooperative_groups::coalesced_threads();

				// If this is not the first set of warps being processed, set the warp group's offset to be the largest result of the previous iteration's interwarp prefix sum
				const T warp_gp_offset = (i == 0) ? 0 : warp_level_num_elems_arr[i - 1];

				T warp_level_num_elems = warp_level_num_elems_arr[i + curr_block.thread_rank()];

				// Interwarp prefix sum: do inclusive prefix sum on warp-level values; calculates total number of elements from beginning of current warp group to currently processed warp
				warp_level_num_elems = cooperative_groups::inclusive_scan(interwarp_active_threads, warp_level_num_elems);

				// Add warp_gp_offset to result, thereby getting total number of elements from beginning of block to currently processed warp; store result in shared memory
				warp_level_num_elems_arr[i + curr_block.thread_rank()] = warp_gp_offset + warp_level_num_elems;
			}
		}
	}

	// Total number of slots needed to store all active metacells processed by block in current iteration is now known and stored in the last element of warp_level_num_elems_arr
	curr_block.sync();


	// Single thread in block allocates space in res_arr_d with atomic operation
	if (curr_block.thread_rank() == 0)
	{
		const T block_level_num_elems = warp_level_num_elems_arr[warps_per_block - 1];
		block_level_res_start_ind = atomicAdd(&res_arr_ind_d, block_level_num_elems);
	}

	// Set an arrival token so that when all threads have passed this point (in particular, the thread with linear ID 0 that allocates memory for the block), it is safe to exit the function (and write to the associated location in memory)
	cooperative_groups::thread_block::arrival_token alloc_block_res_mem_arrival_token = curr_block.barrier_arrive();

	T warp_offset_in_block;	// Calculated with exclusive prefix sum

	// All warps acquire their warp-level offset, which is the element of index (warp ID - 1) in warp_level_num_elems_arr
	// Acquisition of warp-level offset is independent of memory allocation, so can occur anytime that is both after the post-interwarp shuffle curr_block.sync() call and before the writing of results to global memory
	if (curr_block.thread_rank() / warpSize == 0)
		warp_offset_in_block = 0;
	else
		warp_offset_in_block = warp_level_num_elems_arr[curr_block.thread_rank() / warpSize - 1];

	// Make sure all threads (and in particular, the thread with linear ID 0 that allocates memory for the block) have passed the memory-allocation portion of the code before continuing on with exiting the function (and writing to that portion of memory)
	curr_block.barrier_wait(std::move(alloc_block_res_mem_arrival_token));

	// At this point, block-level start index for reporting results guaranteed to be known and saved in shared memory reference variable block_level_res_start_ind

	// Return thread offset in block
	return warp_offset_in_block + thread_offset_in_warp;
};

#endif
