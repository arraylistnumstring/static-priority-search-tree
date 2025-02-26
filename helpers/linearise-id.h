#ifndef LINEARISE_ID_H
#define LINEARISE_ID_H

#include <type_traits>


// For enums, first value (if unspecified) is guaranteed to be 0, and all other unspecified values have value (previous enum's value) + 1
enum Dims { X_DIM_IND, Y_DIM_IND, Z_DIM_IND, NUM_DIMS };


template <typename GridDimType>
	requires std::is_integral<GridDimType>::value
__forceinline__ __host__ __device__ GridDimType lineariseID(const GridDimType x, const GridDimType y,
															const GridDimType z,
															const GridDimType grid_dims_x,
															const GridDimType grid_dims_y)
{
	return x + (y + z * grid_dims_y) * grid_dims_x;
};

__forceinline__ __device__ auto linBlockID()
{
	return lineariseID(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y);
};

__forceinline__ __device__ auto linThreadIDInBlock()
{
	return lineariseID(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y);
};

__forceinline__ __device__ auto linThreadIDInGrid()
{
	return linBlockID() * blockDim.x * blockDim.y * blockDim.z + linThreadIDInBlock();
};

#endif
