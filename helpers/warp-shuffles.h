#ifndef WARP_SHUFFLES_H
#define WARP_SHUFFLES_H

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

// To be applicable to both IPS and PST, logic that is based on number-of-elements boundary conditions has been removed; instead, the entire block does all calculations, with each thread deciding (in the scope that calls calcAllocReportIndOffset()) whether or not to report a result based on its value of cell_active
/*
	When only doing intrawarp shuffles, calcAllocReportIndOffset():
		- Allocates space in the output array; and
		- Returns for each thread the index res_arr_output_ind at which it can safely write its result to the output array.

	When doing intrawarp and interwarp shuffles, calcAllocReportIndOffset():
		- Allocates space in the output array;
		- Stores the first index allocated for the block in *block_level_start_ind_ptr (in a safely concurrent way such that this variable can be a location in shared memory); and
		- Returns for each thread the offset from the first index allocated for the block in the output array, such that a thread can safely write to index (*block_level_start_ind_ptr + block_level_offset) in the output array.

	Pre-conditions:
		Value of interwarp_shfl must be compile-time determinable
		warp_level_num_elems_arr == nullptr iff block_level_start_ind_ptr == nullptr iff interwarp_shfl == false
*/
template <bool interwarp_shfl, typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T calcAllocReportIndOffset(const bool cell_active, T *const warp_level_num_elems_arr = nullptr, T *const block_level_start_ind_ptr = nullptr)
{
	T thread_level_num_elems = cell_active ? 1 : 0;

	// Generate mask for threads active during intrawarp phase; all threads in warp run this (or else are exited, i.e. simply not running any code at all)
	// Call to __ballot_sync() is necessary to determine the thread in warp with largest ID that is still active; this ensures correct delegation of reporting of intrawarp offset results to shared memory
	// As of time of writing (compute capability 9.0), __ballot_sync() returns an unsigned int
	const auto intrawarp_mask = __ballot_sync(0xffffffff, true);

	// Intrawarp prefix sum
	thread_level_num_elems = warpPrefixSum(intrawarp_mask, thread_level_num_elems);

	// Exclusive prefix sum result is simply the element in the preceding index of the result of the inclusive prefix sum; note that as each thread is responsible for at most 1 output element, this is effectively thread_level_num_elems - 1_{cell_active}, where 1_{cell_active} is the indicator function for whether the thread's own value of cell_active is true
	// Use of __shfl_up_sync() in this instance is for generality and for ease of calculation
	T thread_level_offset = __shfl_up_sync(intrawarp_mask, thread_level_num_elems, 1);

	// First thread in each warp read from its own value, so reset that offset
	if (threadIdx.x % warpSize == 0)
		thread_level_offset = 0;

	// If only doing intrawarp shuffle, allocate space in output array on a per-warp basis
	if constexpr (!interwarp_shfl)
	{
		T res_arr_output_ind;

		// Last active thread in warp allocates total number of slots necessary for all active cells in the output array
		if (threadIdx.x % warpSize == fls(intrawarp_mask))
			// Warp-level start index
			res_arr_output_ind = atomicAdd(&res_arr_ind_d, thread_level_num_elems);

		// Broadcast warp-level start index to other threads in warp (i.e. all active threads read from the last active thread in the warp)
		res_arr_output_ind = __shfl_sync(intrawarp_mask, res_arr_output_ind, fls(intrawarp_mask));

		// Add thread-level offset to warp-level start index to find output index for this thread
		res_arr_output_ind += thread_level_offset;

		return res_arr_output_ind;
	}
	else	// if constexpr (interwarp_shfl)
	{
		// Last active thread in warp puts total number of slots necessary for all active cells in the warp_level_num_elems_arr shared memory array
		if (threadIdx.x % warpSize == fls(intrawarp_mask))
			// Place total number of elements in this warp at the slot assigned to this warp in shared memory array warp_level_num_elems_arr
			warp_level_num_elems_arr[threadIdx.x / warpSize] = thread_level_num_elems;

		__syncthreads();	// Warp-level info must be ready to use at the block level

		// Interwarp prefix sum (block-level)
		const auto warps_per_block = blockDim.x * blockDim.y * blockDim.z / warpSize
										+ ((blockDim.x * blockDim.y * blockDim.z % warpSize == 0) ? 0 : 1);

		// Being a prefix sum, only one warp should be active in this process for speed and correctness; use the first warp, which is guaranteed to exist
		if (threadIdx.x / warpSize == 0)
		{
			// If necessary, repeat interwarp prefix sum with base offset of previous iteration until all warps have had their prefix sum calculated
			// Though would be fine to put __ballot_sync() call in loop initialiser here, as there are currently no __syncwarp() calls within the loop that could cause hang, put within loop anyway in case __syncwarp() calls turn out to be necessary in later design
#pragma unroll
			for (auto i = 0; i < warps_per_block; i += warpSize)
			{
				// Simultaneously serves as a __syncwarp() call to ensure that writes and reads are separated, as well as a mask generator for interwarp prefix sum
				const auto interwarp_mask = __ballot_sync(0xffffffff, i + threadIdx.x < warps_per_block);

				// Inter-warp condition
				if (threadIdx.x < warps_per_block)
				{
					// If this is not the first set of warps being processed, set the warp group's offset to be the largest result of the previous iteration's interwarp prefix sum
					T warp_gp_offset = (i == 0) ? 0 : warp_level_num_elems_arr[i - 1];

					T warp_level_num_elems = warp_level_num_elems_arr[i + threadIdx.x];

					// Do inclusive prefix sum on warp-level values
					// Interwarp prefix sum
					warp_level_num_elems = warpPrefixSum(interwarp_mask, warp_level_num_elems);

					// Store result in shared memory, including the effect of the offset
					warp_level_num_elems_arr[i + threadIdx.x] = warp_gp_offset + warp_level_num_elems;
				}
			}
		}

		// Total number of slots needed to store all active metacells processed by block in current iteration is now known and stored in the last element of warp_level_num_elems_arr
		__syncthreads();

		// Single thread in block allocates space in res_arr_d with atomic operation
		if (threadIdx.x == 0)
		{
			const T block_level_num_elems = warp_level_num_elems_arr[warps_per_block - 1];
			*block_level_start_ind_ptr = atomicAdd(&res_arr_ind_d, block_level_num_elems);
		}

		T warp_level_offset;	// Calculated with exclusive prefix sum

		// All warps acquire their warp-level offset, which is the element of index (warp ID - 1) in warp_level_num_elems_arr
		// Acquisition of warp-level offset is independent of memory allocation, so can occur anytime that is both after the post-interwarp shuffle __syncthreads() call and before the writing of results to global memory; placed here for optimisation purposes, as all threads other than thread 0 would otherwise be idle between the post-interwarp shuffle __syncthreads() call and the post-memory allocation __syncthread() call
		if (threadIdx.x / warpSize == 0)
			warp_level_offset = 0;
		else
			warp_level_offset = warp_level_num_elems_arr[threadIdx.x / warpSize - 1];

		// Block-level start index now known and saved in address pointed to by block_level_start_ind_ptr
		__syncthreads();

		// Return block-level offset
		return warp_level_offset + thread_level_offset;
	}
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
#ifdef DEBUG
		printf("About to begin shfl_offset = %u prefix sum\n\n", shfl_offset);
#endif
		// Copies value of variable thread_level_num_elems from thread with lane ID that is shfl_offset less than current thread's lane ID
		// Threads can only receive data from other threads participating in the __shfl_*sync() call
		// Attempting to read from an invalid lane ID or non-participating lane causes a thread to read from its own variable
		U addend = __shfl_up_sync(mask, num, shfl_offset);

		// Only modify num if data came from another thread
		if (linThreadIDInBlock() % warpSize >= shfl_offset)
			num += addend;

#ifdef DEBUG
		printf("Completed shfl_offset = %u prefix sum\n\n", shfl_offset);
#endif
	}

	return num;
}

// Demonstrates same behavior as __reduce_*_sync(), but for arbitrary data type T and arbitrary operation
template <typename T, typename U>
	requires std::conjunction<std::is_integral<T>, std::is_unsigned<T>, std::is_arithmetic<U>>::value
__forceinline__ __device__ U warpReduce(const T mask, U num,
										nvstd::function<U(const U &, const U &)> op)
{
	const T last_active_lane = fls(mask);
	// Start with shfl_offset warpSize / 2, then go downwards in powers of 2, as this order of butterfly reduction guarantees that the first m elements will all have the correct result, for m = \max{2^l : 2^l <= n}
	// The alternative, to go upwards from a shfl_offset of 1, only guarantees correct results for the first (m - k) elements in each group of size m, where k is equal to 2m - n (i.e. the number of elements necessary to reach the next power of 2)
#pragma unroll
	for (T shfl_offset = warpSize / 2; shfl_offset > 0; shfl_offset >>= 1)
	{
#ifdef DEBUG
		printf("About to begin shfl_offset = %u reduction\n\n", shfl_offset);
#endif
		// When the fourth parameter, width, is equal to warpSize (the default value), xor pulls data from the lane with ID equal to (current thread's lane ID XOR shfl_offset), interpreting the result as an int, rather than looking for a particular indexed bit as in mask; this results in threads accessing (own lane ID + shfl_offset) mod warpSize
		U operand = __shfl_xor_sync(mask, num, shfl_offset);

		// Check that value came from a valid thread
		if ((linThreadIDInBlock() % warpSize ^ shfl_offset) <= last_active_lane)
			num = op(operand, num);

#ifdef DEBUG
		printf("Completed shfl_offset = %u reduction\n\n", shfl_offset);
#endif
	}

	// All warps get final result from last active thread
	return num;
}

#endif
