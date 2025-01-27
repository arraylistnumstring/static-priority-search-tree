#ifndef GPU_POWER_OF_2_FUNCS_H
#define GPU_POWER_OF_2_FUNCS_H

template <typename U>
	requires std::unsigned_integral<U>
__forceinline__ __host__ __device__ U expOfLeastPowerOf2(const U num)
{
	unsigned exp = 0;
	// Repeat loop until exp is sufficiently large that 2^exp >= num
	while (!(1 << exp >= num))
		exp++;
	return exp;
};

template <typename U>
	requires std::unsigned_integral<U>
__forceinline__ __host__ __device__ U expOfNextGreaterPowerOf2(const U num)
{
	/*
		Smallest power of 2 greater than num is equal to 2^ceil(lg(num + 1))
		ceil(lg(num + 1)) is equal to the number of right bitshifts necessary to make num = 0 (after integer truncation); this method of calcalation is used in order to prevent imprecision of float conversion from causing excessively large (and therefore incorrect) returned integer values
	*/
	unsigned exp = 0;
	while (num >> exp != 0)
		exp++;
	return exp;
};

// Helper function for calculating the smallest power of 2 greater than or equal to num
template <typename U>
	requires std::unsigned_integral<U>
__forceinline__ __host__ __device__ U leastPowerOf2(const U num)
{
	return 1 << expOfLeastPowerOf2(num);
};

// Helper function for calculating the next power of 2 greater than num
template <typename U>
	requires std::unsigned_integral<U>
__forceinline__ __host__ __device__ U nextGreaterPowerOf2(const U num)
{
	return 1 << expOfNextGreaterPowerOf2(num);
};


#endif
