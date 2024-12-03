#ifndef ISOSURFACE_DATA_PROCESSING_H
#define ISOSURFACE_DATA_PROCESSING_H

#include <fstream>
#include <iostream>
#include <type_traits>

#include "class-member-checkers.h"
#include "dev-symbols.h"
#include "exit-status-codes.h"
#include "gpu-err-chk.h"

// For enums, first value (if unspecified) is guaranteed to be 0, and all other unspecified values have value (previous enum's value) + 1
enum Dims { X_DIM_IND, Y_DIM_IND, Z_DIM_IND, NUM_DIMS };

// Forward declarations
template <typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
__forceinline__ __device__ void getVoxelMinMax(T *const vertex_arr_d, T &min, T &max,
									const GridDimType base_voxel_coord_x,
									const GridDimType base_voxel_coord_y,
									const GridDimType base_voxel_coord_z,
									const GridDimType pt_grid_dims_x,
									const GridDimType pt_grid_dims_y
									);

template <typename GridDimType>
	requires std::is_integral<GridDimType>::value
// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
__forceinline__ __host__ __device__
#else
inline
#endif
// grid_dims split into 3 separate variables for compatability with on-device code
GridDimType lineariseID(const GridDimType x, const GridDimType y, const GridDimType z,
						const GridDimType grid_dims_x, const GridDimType grid_dims_y);


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
	// Use of decltype to allow for appropriate instantiation of template function std::min (as implicit casting does not take place among its parameters)
	dim3 num_blocks(std::min(static_cast<decltype(MAX_X_DIM_NUM_BLOCKS)>(metacell_grid_dims[Dims::X_DIM_IND]),
								MAX_X_DIM_NUM_BLOCKS),
					std::min(static_cast<decltype(MAX_Y_DIM_NUM_BLOCKS)>(metacell_grid_dims[Dims::Y_DIM_IND]),
								MAX_Y_DIM_NUM_BLOCKS),
					std::min(static_cast<decltype(MAX_Z_DIM_NUM_BLOCKS)>(metacell_grid_dims[Dims::Z_DIM_IND]),
								MAX_Z_DIM_NUM_BLOCKS)
					);
	dim3 threads_per_block(std::min(static_cast<decltype(MAX_X_DIM_THREADS_PER_BLOCK)>(metacell_dims[Dims::X_DIM_IND]),
										MAX_X_DIM_THREADS_PER_BLOCK),
							std::min(static_cast<decltype(MAX_Y_DIM_THREADS_PER_BLOCK)>(metacell_dims[Dims::Y_DIM_IND]),
										MAX_Y_DIM_THREADS_PER_BLOCK),
							std::min(static_cast<decltype(MAX_Z_DIM_THREADS_PER_BLOCK)>(metacell_dims[Dims::Z_DIM_IND]),
										MAX_Z_DIM_THREADS_PER_BLOCK)
							);

	/*
		{} used here as array initialiser notation; based on Stack Overflow test case, this method is superior from a logical organisation perspective and likely also from a performance perspective, despite requiring storage in global memory
		Source:
			https://stackoverflow.com/a/65064081
	*/
	formMetacellsGlobal<<<num_blocks, threads_per_block/*, Some shared memory for inter-warp reduces(?) */>>>(vertex_arr_d, metacell_arr_d,
					pt_grid_dims[Dims::X_DIM_IND], pt_grid_dims[Dims::Y_DIM_IND], pt_grid_dims[Dims::Z_DIM_IND],
					metacell_grid_dims[Dims::X_DIM_IND], metacell_grid_dims[Dims::Y_DIM_IND], metacell_grid_dims[Dims::Z_DIM_IND],
					metacell_dims[Dims::X_DIM_IND], metacell_dims[Dims::Y_DIM_IND], metacell_dims[Dims::Z_DIM_IND]
				);

	return metacell_arr_d;
};

// Must have point grid and metacell grid dimension values available, in case there is a discrepancy between the maximal possible thread block size and actual metacell size; and/or maximal possible thread grid size and actual metacell grid size
// To minimise global memory access time (and because the number of objects passed is relatively small for each set of dimensions), use explicitly passed scalar parameters for each dimension
template <typename PointStruct, typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
__global__ void formMetacellsGlobal(T *const vertex_arr_d, PointStruct *const metacell_arr_d,
										GridDimType pt_grid_dims_x, GridDimType pt_grid_dims_y, GridDimType pt_grid_dims_z,
										GridDimType metacell_grid_dims_x, GridDimType metacell_grid_dims_y, GridDimType metacell_grid_dims_z,
										GridDimType metacell_dims_x, GridDimType metacell_dims_y, GridDimType metacell_dims_z
									)
{

	T min_vert_val, max_vert_val;
	// Repeat over entire voxel grid
	// Data is z-major, then y-major, then x-major (i.e. x-dimension index changes the fastest, followed by y-index, then z-index)
#pragma unroll
	for (GridDimType k = 0; k < pt_grid_dims_z; k += gridDim.z * blockDim.z)
	{
#pragma unroll
		for (GridDimType j = 0; j < pt_grid_dims_y; j += gridDim.y * blockDim.y)
		{
#pragma unroll
			for (GridDimType i = 0; i < pt_grid_dims_x; i += gridDim.x * blockDim.x)
			{
				// Check that voxel in question exists
				GridDimType base_voxel_coord_x = i + blockIdx.x * blockDim.x + threadIdx.x;
				GridDimType base_voxel_coord_y = j + blockIdx.y * blockDim.y + threadIdx.y;
				GridDimType base_voxel_coord_z = k + blockIdx.z * blockDim.z + threadIdx.z;
				// One voxel is associated with each vertex, with the exception of the last vertices in each dimension (i.e. those vertices with at least one coordinate of value pt_grid_dims_[x-z] - 1)
				if (base_voxel_coord_x < pt_grid_dims_x - 1
						&& base_voxel_coord_y < pt_grid_dims_y - 1
						&& base_voxel_coord_z < pt_grid_dims_z - 1)
				{
					getVoxelMinMax(vertex_arr_d, min_vert_val, max_vert_val,
										base_voxel_coord_x, base_voxel_coord_y, base_voxel_coord_z,
										pt_grid_dims_x, pt_grid_dims_y);
				}

				// Intrawarp shuffle for metacell min-max val determination
				// Interwarp shuffle for metacell min-max val determination

				// Single thread in block writes result to global memory array
				if (threadIdx.x == 0)
				{
					// Cast necessary, as an arithemtic operation (even of two types that are both small, e.g. GridDimType = char) effects an up-casting to a datatype at least as large as int, whereas directly supplied variables remain as the previous type, causing the overall template instantiation of lineariseID to fail
					GridDimType metacellID = lineariseID(static_cast<GridDimType>(base_voxel_coord_x / metacell_dims_x),
															static_cast<GridDimType>(base_voxel_coord_y / metacell_dims_y),
															static_cast<GridDimType>(base_voxel_coord_z / metacell_dims_z),
															metacell_grid_dims_x,
															metacell_grid_dims_y
														);
					metacell_arr_d[metacellID].dim1_val = min_vert_val;
					metacell_arr_d[metacellID].dim2_val = max_vert_val;
					if constexpr (HasID<PointStruct>::value)
						metacell_arr_d[metacellID].id = metacellID;
				}
			}
		}
	}
};

template <typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
__forceinline__ __device__ void getVoxelMinMax(T *const vertex_arr_d, T &min, T &max,
									const GridDimType base_voxel_coord_x,
									const GridDimType base_voxel_coord_y,
									const GridDimType base_voxel_coord_z,
									const GridDimType pt_grid_dims_x,
									const GridDimType pt_grid_dims_y
								)
{
	// Each thread accesses the vertex of its voxel with the lowest indices in each dimension and uses this scalar value as the initial value with which future values are compared
	max = min = vertex_arr_d[lineariseID(base_voxel_coord_x, base_voxel_coord_y, base_voxel_coord_z, pt_grid_dims_x, pt_grid_dims_y)];

	// Check each vertex of the voxel and get the maximum and minimum values achieved at those vertices
	for (GridDimType k = 0; k < 2; k++)
	{
		for (GridDimType j = 0; j < 2; j++)
		{
			for (GridDimType i = 0; i < 2; i++)
			{
				T curr_vert = vertex_arr_d[lineariseID(static_cast<GridDimType>(base_voxel_coord_x + i),
															static_cast<GridDimType>(base_voxel_coord_y + j),
															static_cast<GridDimType>(base_voxel_coord_z + k),
															pt_grid_dims_x, pt_grid_dims_y
														)];
				min = min <= curr_vert ? min : curr_vert;
				max = max >= curr_vert ? max : curr_vert;
			}
		}
	}
}

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
