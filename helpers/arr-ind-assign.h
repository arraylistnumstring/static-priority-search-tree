#ifndef ARR_IND_ASSIGN_H
#define ARR_IND_ASSIGN_H

// Assigning elements of an array on device such that array[i] = i
// No shared memory usage
__global__ void arrIndAssign(size_t *const ind_arr, const size_t num_elems)
{
	// Simple iteration over entire array, instantiating each array element with the value of its index; no conflicts possible, so no synchronisation necessary
	// Use loop unrolling to decrease register occupation, as the number of loops is known when kernel is called
#pragma unroll
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
			i < num_elems; i += gridDim.x * blockDim.x)
		ind_arr[i] = i;
}

#endif
