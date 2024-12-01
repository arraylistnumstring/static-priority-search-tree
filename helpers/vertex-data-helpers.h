#ifndef VERTEX_DATA_HELPERS_H
#define VERTEX_DATA_HELPERS_H

template <typename GridDimType>
// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
__forceinline__ __host__ __device__
#else
inline
#endif
GridDimType linearVertID(const GridDimType x, const GridDimType y, const GridDimType z, GridDimType const vert_grid_dims[NUM_DIMS])
{
	return x + (y + z * vert_grid_dims[1]) * vert_grid_dims[0];
};

#endif
