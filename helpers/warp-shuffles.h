#ifndef WARP_SHUFFLES_H
#define WARP_SHUFFLES_H

// Helper function fls() for "find last (bit) set", where bits are indexed from least to most significant place value
template <typename T>
	requires std::unsigned_integral<T>
__forceinline__ __device__ T fls(T val)	// Equivalent to truncate(log_2(val))
{
	T bit_ind = 0;
	while (val >>= 1)	// Unsigned right shifts are logical, i.e. zero-filling; loop exits when val evaluates to 0 after the right shift
		bit_ind++;
	return bit_ind;
};

// Calculates inclusive prefix sum using thread-local variables and intra-warp shuffle; returns result in reference variable num
template <typename T>
	requires std::is_arithmetic<T>::value
__forceinline__ __device__ void warpPrefixSum(const unsigned mask, T &num)
{
	// As a comma-delimited declaration does not allow declarations of different types, create the const-valued shfl_offset_lim before the loop instead
	// fls() will always be at most warpSize - 1 (32 bits, with the leftmost 0-indexed as bit 31), so shfl_offset_lim will evaluate to warpSize if the entire warp is active
	const unsigned shfl_offset_lim = fls(intrawarp_mask) + 1;

	// Parallel prefix sum returns accurate values as long as the loop runs for every shift size (shfl_offset) that is a power of 2 (with nonnegative exponent) up to the largest power of 2 less than shfl_offset_lim (the latter is at most warpSize)
#pragma unroll
	for (unsigned shfl_offset = 1; shfl_offset < shfl_offset_lim; shfl_offset <<= 1)
	{
		// Copies value of variable thread_level_num_elems from thread with lane ID that is shfl_offset less than current thread's lane ID
		// Threads can only receive data from other threads participating in the __shfl_*sync() call
		// Attempting to read from an invalid lane ID or non-participating lane causes a thread to read from its own variable
		T addend = __shfl_up_sync(mask, num, shfl_offset);

		// Only modify num if data came from another thread
		if (threadIdx.x % warpSize >= shfl_offset)
			num += addend;
	}
}

#endif
