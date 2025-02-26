#ifndef POWER_OF_2_FUNCS_H
#define POWER_OF_2_FUNCS_H

#include <concepts>

// Generates exponent for maximal power of 2 less than or equal to num
template <typename U>
	requires std::unsigned_integral<U>
// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
U expOfMaxPowerOf2AtMost(const U num)
{
	unsigned exp = 0;
	// Increment exp until the first instance where 2^exp > num
	while (1 << exp <= num)
		exp++;
	// Return exp - 1
	return exp - 1;
};

// Generates exponent for minimal power of 2 greater than or equal to num
template <typename U>
	requires std::unsigned_integral<U>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
U expOfMinPowerOf2AtLeast(const U num)
{
	unsigned exp = 0;
	// Repeat loop until exp is sufficiently large that 2^exp >= num
	while (!(1 << exp >= num))
		exp++;
	return exp;
};

// Generates exponent for minimal power of 2 greater than num
template <typename U>
	requires std::unsigned_integral<U>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
U expOfMinPowerOf2GreaterThan(const U num)
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

// Helper function for calculating the largest power of 2 less than or equal to num
template <typename U>
	requires std::unsigned_integral<U>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
U maxPowerOf2AtMost(const U num)
{
	if (num == 0)
		return 0;
	else
		return 1 << expOfMaxPowerOf2AtMost(num);
};

// Helper function for calculating the smallest power of 2 greater than or equal to num
template <typename U>
	requires std::unsigned_integral<U>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
U minPowerOf2AtLeast(const U num)
{
	if (num == 0)
		return 0;
	else
		return 1 << expOfMinPowerOf2AtLeast(num);
};

// Helper function for calculating the next power of 2 greater than num
template <typename U>
	requires std::unsigned_integral<U>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
U minPowerOf2GreaterThan(const U num)
{
	return 1 << expOfMinPowerOf2GreaterThan(num);
};


#endif
