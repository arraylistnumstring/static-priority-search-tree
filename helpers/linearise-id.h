#ifndef LINEARISE_ID_H
#define LINEARISE_ID_H

template <typename GridDimType>
	requires std::is_integral<GridDimType>::value
// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
__forceinline__ __host__ __device__
#else
inline
#endif
GridDimType lineariseID(const GridDimType x, const GridDimType y, const GridDimType z,
						GridDimType grid_dims_x, const GridDimType grid_dims_y)
{
	return x + (y + z * grid_dims_y) * grid_dims_x;
}

#endif
