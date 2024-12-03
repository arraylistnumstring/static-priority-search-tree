#ifndef ISOSURFACE_DATA_PROCESSING_H
#define ISOSURFACE_DATA_PROCESSING_H

#include <fstream>
#include <iostream>
#include <type_traits>

#include "dev-symbols.h"
#include "exit-status-codes.h"
#include "gpu-err-chk.h"

// For enums, first value (if unspecified) is guaranteed to be 0, and all other unspecified values have value (previous enum's value) + 1
enum Dims { X_DIM_IND, Y_DIM_IND, Z_DIM_IND, NUM_DIMS };

template <template<typename, typename, size_t> class PointStructTemplate,
		 	size_t num_IDs, typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
PointStructTemplate<T, GridDimType, num_IDs> *formMetacells(T *const vertex_arr_d, GridDimType pt_grid_dims[Dims::NUM_DIMS], GridDimType metacell_dims[Dims::NUM_DIMS], size_t &num_metacells, const int dev_ind, const int num_devs)
{
	// Total number of metacells is \Pi_{i=1}^Dims::NUM_DIMS ceil((pt_grid_dims[i] - 1) / metacell_dims[i])
	// Note that the ceiling function is used because if metacell_dims[i] \not | (pt_grid_dims[i] - 1), then the last metacell(s) in dimension i will be nonempty, though not fully tiled
	// The -1 addend arises because if one surjectively assigns to each metacell the vertex on its volume that has the smallest indices in each dimension, the edges of the point grid where indices are largest in a given direction will have no metacells, as there are no further points to which to interpolate or draw such a metacell
	GridDimType metacell_grid_dims[NUM_DIMS];
	num_metacells = 1;
	for (int i = 0; i < Dims::NUM_DIMS; i++)
	{
		metacell_grid_dims[i] = (pt_grid_dims[i] - 1) / metacell_dims[i]
								+ ( (pt_grid_dims[i] - 1) % metacell_dims[i] == 0 ? 0 : 1);
		num_metacells *= metacell_grid_dims[i];
	}

	// On-device memory allocations for metacell array
	PointStructTemplate<T, GridDimType, num_IDs> *metacell_arr_d;

	gpuErrorCheck(cudaMalloc(&metacell_arr_d, num_metacells * sizeof(PointStructTemplate<T, GridDimType, num_IDs>)),
					"Error in allocating metacell storage array on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");

	// Set grid size to be equal to number of metacells, unless this exceeds the GPU's capabilities, as determined by its compute capability-associated technical specifications
	dim3 num_blocks(std::min(metacell_grid_dims[Dims::X_DIM_IND], MAX_X_DIM_NUM_BLOCKS),
					std::min(metacell_grid_dims[Dims::Y_DIM_IND], MAX_Y_DIM_NUM_BLOCKS),
					std::min(metacell_grid_dims[Dims::Z_DIM_IND], MAX_Z_DIM_NUM_BLOCKS)
					);
	dim3 threads_per_block(std::min(metacell_dims[Dims::X_DIM_IND], MAX_X_DIM_THREADS_PER_BLOCK),
							std::min(metacell_dims[Dims::Y_DIM_IND], MAX_Y_DIM_THREADS_PER_BLOCK),
							std::min(metacell_dims[Dims::Z_DIM_IND], MAX_Z_DIM_THREADS_PER_BLOCK)
							);

	/*
		{} used here as array initialiser notation; based on Stack Overflow test case, this method is superior from a logical organisation perspective and likely also from a performance perspective, despite requiring storage in global memory
		Source:
			https://stackoverflow.com/a/65064081
	*/
	formMetacellsGlobal<<<num_blocks, threads_per_block/*, Some shared memory for inter-warp reduces(?) */>>>(vertex_arr_d, metacell_arr_d,
					{pt_grid_dims[Dims::X_DIM_IND], pt_grid_dims[Dims::Y_DIM_IND], pt_grid_dims[Dims::Z_DIM_IND]},
					{metacell_grid_dims[Dims::X_DIM_IND], metacell_grid_dims[Dims::Y_DIM_IND], metacell_grid_dims[Dims::Z_DIM_IND]},
					{metacell_dims[Dims::X_DIM_IND], metacell_dims[Dims::Y_DIM_IND], metacell_dims[Dims::Z_DIM_IND]}
				);

	return metacell_arr_d;
};

// Must have point grid and metacell grid dimension values available, in case there is a discrepancy between the maximal possible thread block size and actual metacell size; and/or maximal possible thread grid size and actual metacell grid size
template <typename PointStruct, typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
__global__ void formMetacellsGlobal(T *const vertex_arr_d, PointStruct *const metacell_arr_d,
									GridDimType pt_grid_dims[Dims::NUM_DIMS],
									GridDimType metacell_grid_dims[Dims::NUM_DIMS],
									GridDimType metacell_dims[Dims::NUM_DIMS])
{
	/*
	T min_vert_val, max_vert_val;
	// Repeat over entire metacell grid
	// Data is z-major, then y-major, then x-major (i.e. x-dimension index changes the fastest, followed by y-index, then z-index)
	for (GridDimType k = 0; k < metacell_grid_dims[Dims::Z_DIM_IND]; k += gridDim.z)
	{
		for (GridDimType j = 0; j < metacell_grid_dims[Dims::Y_DIM_IND]; j += gridDim.y)
		{
			for (GridDimType i = 0; i < metacell_grid_dims[Dims::X_DIM_IND]; i += gridDim.x)
			{
				getVoxelMinMax(vertex_arr_d, min_vert_val, max_vert_val,
									// Array initialiser notation
									{
										i * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x,
									}
								);
			}
		}
	}
};

template <typename T, typename GridDimType>
__device__ void getVoxelMinMax(T *const vertex_arr_d, T &min, T &max, GridDimType grid_offset[Dims::NUM_DIMS])
{
	// Each thread accesses the 
	max = min = vertex_arr_d[lineariseID(grid_offset[Dims::X_DIM_IND] * gridDim.x * blockDim.x
												+ blockIdx.x * blockDim.x + threadIdx.x,
											grid_offset[Dims::Y_DIM_IND] * gridDim.y * blockDim.y
												+ blockIdx.y * blockDim.y + threadIdx.y,
											grid_offset[Dims::Z_DIM_IND] * gridDim.z * blockDim.z
												+ blockIdx.z * blockDim.z + threadIdx.z,
											)
							];
	// Check each vertex of the voxel and get the maximum and minimum values achieved at those vertices
	for (GridDimType k = 0; k < 2; k++)
		for (GridDimType j = 0; j < 2; j++)
			for (GridDimType i = 0; i < 2; i++)
			{

			}
			*/
};

template <typename GridDimType>
// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
__forceinline__ __host__ __device__
#else
inline
#endif
GridDimType lineariseID(const GridDimType x, const GridDimType y, const GridDimType z, GridDimType const grid_dims[Dims::NUM_DIMS])
{
	return x + (y + z * grid_dims[Dims::Y_DIM_IND]) * grid_dims[Dims::X_DIM_IND];
};

template <typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
T *readInVertices(std::string input_filename, GridDimType num_vertices)
{
	T *vertex_arr = new T[num_vertices];

	std::ifstream input_filestream(input_filename, std::ios_base::binary);

	if (!input_filestream.is_open())	// Check that filestream is open
	{
		std::cerr << "Failed to open " << input_filename << " for reading\n";
		std::exit(ExitStatusCodes::FILE_OPEN_ERR);
	}

	input_filestream.read(reinterpret_cast<char *>(vertex_arr), num_vertices * sizeof(T));

	// Check that the appropriate number of bytes were read
	if (input_filestream.gcount() != num_vertices * sizeof(T))
	{
		std::cerr << "Failure: read " << input_filestream.gcount() << " bytes, expected " << num_vertices * sizeof(T) << " bytes\n";
		std::exit(ExitStatusCodes::INPUT_READ_ERR);
	}

	return vertex_arr;
};

#endif
