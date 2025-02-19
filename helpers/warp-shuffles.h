#ifndef WARP_SHUFFLES_H
#define WARP_SHUFFLES_H

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
// Allows use of nvstd::function, an equivalent to std::function that functions on both host and device (but not across the host-device boundary)
#include <nvfunctional>

#include "dev-symbols.h"	// Need access to __device__ (global memory) variable res_arr_ind_d
#include "linearise-id.h"

// Forward declarations
template <typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T fls(T val);

template <typename T, typename U>
	requires std::conjunction<
								std::is_integral<T>,
								std::is_unsigned<T>,
								std::is_arithmetic<U>
			>::value
__forceinline__ __device__ U warpPrefixSum(const T mask, U num);

// Intrawarp version: allocates space in output array and returns for each thread the index res_arr_output_ind at which it can safely write its result to the output array
template <typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T calcAllocReportIndOffset(T thread_level_num_elems)
{
	// Intrawarp cooperative group
	const cooperative_groups::coalesced_group intrawarp_active_threads
			= cooperative_groups::coalesced_threads();

	// Exclusive prefix sum result is simply the element in the preceding index of the result of the inclusive prefix sum; note that as each thread is responsible for at most 1 output element, this is effectively thread_level_num_elems - 1_{active_data}, where 1_{active_data} is the indicator function for whether the thread's own value of active_data is true
	T thread_level_offset = cooperative_groups::exclusive_scan(intrawarp_active_threads, thread_level_num_elems);

	thread_level_num_elems = cooperative_groups::inclusive_scan(intrawarp_active_threads, thread_level_num_elems);

	// Intrawarp: allocate space in output array on per-warp basis
	T res_arr_output_ind;

	// Last thread in coalesced group allocates total number of slots necessary for all active cells in the output array
	const auto last_thread_in_group = intrawarp_active_threads.num_threads() - 1;
	if (intrawarp_active_threads.thread_rank() == last_thread_in_group)
		// Warp-level start index
		res_arr_output_ind = atomicAdd(&res_arr_ind_d, thread_level_num_elems);

	// Broadcast warp-level start index to other threads in warp (i.e. all active threads read from last active thread in the warp)
	res_arr_output_ind = intrawarp_active_threads.shfl(res_arr_output_ind, last_thread_in_group);

	// Add thread-level offset to warp-level start index to find output index for this thread
	res_arr_output_ind += thread_level_offset;

	return res_arr_output_ind;
}

// Intrawarp + interwarp version: allocates space in output arra; stores first index allocated for the block in warp_level_num_elems_arr
template <typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T calcAllocReportIndOffset(const cooperative_groups::thread_block &curr_block, T thread_level_num_elems, T *const warp_level_num_elems_arr, T &block_level_start_ind)
{
	// Intrawarp cooperative group
	const cooperative_groups::coalesced_group intrawarp_active_threads
			= cooperative_groups::coalesced_threads();

	// Exclusive prefix sum result is simply the element in the preceding index of the result of the inclusive prefix sum; note that as each thread is responsible for at most 1 output element, this is effectively thread_level_num_elems - 1_{active_data}, where 1_{active_data} is the indicator function for whether the thread's own value of active_data is true
	T thread_level_offset = cooperative_groups::exclusive_scan(intrawarp_active_threads, thread_level_num_elems);

	thread_level_num_elems = cooperative_groups::inclusive_scan(intrawarp_active_threads, thread_level_num_elems);

	// Last active thread in warp puts total number of slots necessary for all active cells in the warp_level_num_elems_arr shared memory array
	if (intrawarp_active_threads.thread_rank() == intrawarp_active_threads.num_threads() - 1)
		// Place total number of elements in this warp at the slot assigned to this warp in shared memory array warp_level_num_elems_arr
		warp_level_num_elems_arr[curr_block.thread_rank() / warpSize] = thread_level_num_elems;

	curr_block.sync();		// Warp-level info must be ready to use at the block level


	// Interwarp prefix sum (block-level)
	const auto warps_per_block = curr_block.num_threads() / warpSize
									+ ((curr_block.num_threads() % warpSize == 0) ? 0 : 1);

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

				// Interwarp prefix sum: do inclusive prefix sum on warp-level values
				warp_level_num_elems = cooperative_groups::inclusive_scan(interwapr_active_threads, warp_level_num_elems);

				// Store result in shared memory, including the effect of the offset
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
		*block_level_start_ind = atomicAdd(&res_arr_ind_d, block_level_num_elems);
	}

	T warp_level_offset;	// Calculated with exclusive prefix sum

	// All warps acquire their warp-level offset, which is the element of index (warp ID - 1) in warp_level_num_elems_arr
	// Acquisition of warp-level offset is independent of memory allocation, so can occur anytime that is both after the post-interwarp shuffle __syncthreads() call and before the writing of results to global memory; placed here for optimisation purposes, as all threads other than thread 0 would otherwise be idle between the post-interwarp shuffle __syncthreads() call and the post-memory allocation __syncthread() call
	if (curr_block.thread_rank() / warpSize == 0)
		warp_level_offset = 0;
	else
		warp_level_offset = warp_level_num_elems_arr[curr_block.thread_rank() / warpSize - 1];

	// Block-level start index now known and saved in address pointed to by block_level_start_ind
	curr_block.sync();

	// Return block-level offset
	return warp_level_offset + thread_level_offset;
};

// Helper function fls() for "find last (bit) set", where bits are 0-indexed from least to most significant place value
template <typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T fls(T val)	// Equivalent to truncate(log_2(val))
{
	T bit_ind = 0;
	while (val >>= 1)	// Unsigned right shifts are logical, i.e. zero-filling; loop exits when val evaluates to 0 after the right shift
		bit_ind++;
	return bit_ind;
}

// Calculates inclusive prefix sum using thread-local variables and intra-warp shuffle; returns result in reference variable num
//	Precondition: inactive threads, if any, are located in one contiguous group that ends at the highest-indexed thread (when ordering by increasing linearised ID)
template <typename T, typename U>
	requires std::conjunction<
								// As std::unsigned_integral<T> evaluates to true/false instead of a type name, instead use its constituent parts separately to achieve the same effect
								std::is_integral<T>,
								std::is_unsigned<T>,
								std::is_arithmetic<U>
			>::value
__forceinline__ __device__ U warpPrefixSum(const T mask, U num)
{
	// As a comma-delimited declaration does not allow declarations of different types (including variables of different const-ness), create the const-valued shfl_offset_lim outside of the loop initialiser, i.e. before shfl_offset
	// At time of writing (compute capability 9.0), warp size is 32; hence, fls() will always be at most warpSize - 1 (32 bits, with the leftmost 0-indexed as bit 31), so shfl_offset_lim will evaluate to warpSize if the entire warp is active
	const T shfl_offset_lim = fls(mask) + 1;

	// Parallel prefix sum returns accurate values as long as the loop runs for every shift size (shfl_offset) that is a power of 2 (with nonnegative exponent) up to the largest power of 2 less than shfl_offset_lim (the latter is at most warpSize)
#pragma unroll
	for (T shfl_offset = 1; shfl_offset < shfl_offset_lim; shfl_offset <<= 1)
	{
#ifdef DEBUG_SHFL
		printf("About to begin shfl_offset = %u prefix sum\n\n", shfl_offset);
#endif
		// Copies value of variable thread_level_num_elems from thread with lane ID that is shfl_offset less than current thread's lane ID
		// Threads can only receive data from other threads participating in the __shfl_*sync() call
		// Attempting to read from an invalid lane ID or non-participating lane causes a thread to read from its own variable
		U addend = __shfl_up_sync(mask, num, shfl_offset);

		// Only modify num if data came from another thread
		if (linThreadIDInBlock() % warpSize >= shfl_offset)
			num += addend;

#ifdef DEBUG_SHFL
		printf("Completed shfl_offset = %u prefix sum\n\n", shfl_offset);
#endif
	}

	return num;
}

#endif
