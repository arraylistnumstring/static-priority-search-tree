#ifndef ATOMICS_H
#define ATOMICS_H

template <typename T>
	requires std::unsigned_integral<T>
__device__ T atomicIncNoWrap(T *addr, T inclusive_limit)
{
	// Declarations are guaranteed to evaluate from left to right, with all declarators to the left of a sequence point (e.g. a comma or a semicolon) completing before the next declarator (C++ specification, section 6.7, 6.7.6)
	// Sentinel value of old - 1 used for initialisation of assumed to allow entering of while loop
	T old = *addr, assumed = old - 1;

	// Attempt to modify *addr only if current value in variable has not already hit the indicated maximum
	// assumed != old iff *addr has not been updated, as atomicCAS assigns old + 1 to *addr iff assumed == value of *addr at function call time, then returns the value of *addr at function call time
	while (old < inclusive_limit && assumed != old)
	{
		assumed = old;
		old = atomicCAS(addr, assumed, old + 1);
	}

	return old;
};

template <typename T>
	requires std::unsigned_integral<T>
__device__ T atomicDecNoWrap(T *addr, T inclusive_limit)
{
	// Sentinel value of old + 1 used for initialisation of assumed to allow entering of while loop
	T old = *addr, assumed = old + 1;

	// Attempt to modify *addr only if current value in variable has not already hit the indicated maximum
	// assumed != old iff *addr has not been updated, as atomicCAS assigns old - 1 to *addr iff assumed == value of *addr at function call time, then returns the value of *addr at function call time
	while (old > inclusive_limit && assumed != old)
	{
		assumed = old;
		old = atomicCAS(addr, assumed, old - 1);
	}

	return old;
};

#endif
