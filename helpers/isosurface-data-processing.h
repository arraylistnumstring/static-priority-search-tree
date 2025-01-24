#ifndef ISOSURFACE_DATA_PROCESSING_H
#define ISOSURFACE_DATA_PROCESSING_H

#include <fstream>
#include <iostream>
// Allows use of nvstd::function, an equivalent to std::function that functions on both host and device (but not across the host-device boundary)
#include <nvfunctional>
#include <type_traits>

#include "class-member-checkers.h"
#include "data-size-concepts.h"
#include "dev-symbols.h"
#include "exit-status-codes.h"
#include "gpu-err-chk.h"
#include "linearise-id.h"
#include "warp-shuffles.h"


// Calculates metacell grid dimensions and total number of metacells, returning them using the supplied pointer and reference parameters, calculated from inputs of vertex grid dimensions and metacell dimensions
template <typename GridDimType>
	requires std::is_integral<GridDimType>::value
void calcNumMetacells(GridDimType pt_grid_dims[Dims::NUM_DIMS], GridDimType metacell_dims[Dims::NUM_DIMS],
						GridDimType metacell_grid_dims[Dims::NUM_DIMS], size_t &num_metacells)
{
	// Total number of metacells is \prod_{i=1}^Dims::NUM_DIMS ceil((pt_grid_dims[i] - 1) / metacell_dims[i])
	// Note that the ceiling function is used because if metacell_dims[i] \not | (pt_grid_dims[i] - 1), then the last metacell(s) in dimension i will be nonempty, though not fully tiled
	// The -1 addend arises because if one surjectively assigns to each metacell the vertex on its volume that has the smallest indices in each dimension, the edges of the point grid where indices are largest in a given direction will have no metacells, as there are no further points to which to interpolate or draw such a metacell
	num_metacells = 1;
	for (int i = 0; i < Dims::NUM_DIMS; i++)
	{
		metacell_grid_dims[i] = (pt_grid_dims[i] - 1) / metacell_dims[i]
								+ ( (pt_grid_dims[i] - 1) % metacell_dims[i] == 0 ? 0 : 1);
		num_metacells *= metacell_grid_dims[i];
	}
};


template <bool interwarp_reduce, typename PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
__global__ void formMetacellTagsGlobal(T *const vertex_arr_d, PointStruct *const metacell_arr_d,
										const GridDimType warps_per_block,
										GridDimType pt_grid_dims_x, GridDimType pt_grid_dims_y, GridDimType pt_grid_dims_z,
										GridDimType metacell_grid_dims_x, GridDimType metacell_grid_dims_y, GridDimType metacell_grid_dims_z,
										GridDimType metacell_dims_x, GridDimType metacell_dims_y, GridDimType metacell_dims_z
									);

template <typename PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
__global__ void formVoxelTagsGlobal(T *const vertex_arr_d, PointStruct *const voxel_arr_d,
										GridDimType pt_grid_dims_x, GridDimType pt_grid_dims_y, GridDimType pt_grid_dims_z
									);

template <typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
__forceinline__ __device__ void getVoxelMinMax(T *const vertex_arr_d, T &min, T &max,
													const GridDimType base_voxel_coord_x,
													const GridDimType base_voxel_coord_y,
													const GridDimType base_voxel_coord_z,
													const GridDimType pt_grid_dims_x,
													const GridDimType pt_grid_dims_y,
													const GridDimType pt_grid_dims_z
												);


// Precondition: number of metacells is in reference num_metacells
template <class PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
PointStruct *formMetacellTags(T *const vertex_arr_d, GridDimType pt_grid_dims[Dims::NUM_DIMS],
								GridDimType metacell_dims[Dims::NUM_DIMS],
								GridDimType metacell_grid_dims[NUM_DIMS], const size_t num_metacells,
								const int dev_ind, const int num_devs, const int warp_size)
{
	// On-device memory allocations for metacell array
	PointStruct *metacell_arr_d;

	gpuErrorCheck(cudaMalloc(&metacell_arr_d, num_metacells * sizeof(PointStruct)),
					"Error in allocating metacell tag storage array on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	// Set grid size to be equal to number of metacells, unless this exceeds the GPU's capabilities, as determined by its compute capability-associated technical specifications
	// Use of decltype to allow for appropriate instantiation of template function std::min (as implicit casting does not take place among its parameters); preprocessor constants here are either sufficiently large to hold GridDimType or at least to avoid type narrowing warnings
	dim3 num_blocks(std::min(static_cast<decltype(MAX_X_DIM_NUM_BLOCKS)>(metacell_grid_dims[Dims::X_DIM_IND]),
								MAX_X_DIM_NUM_BLOCKS),
					std::min(static_cast<decltype(MAX_Y_DIM_NUM_BLOCKS)>(metacell_grid_dims[Dims::Y_DIM_IND]),
								MAX_Y_DIM_NUM_BLOCKS),
					std::min(static_cast<decltype(MAX_Z_DIM_NUM_BLOCKS)>(metacell_grid_dims[Dims::Z_DIM_IND]),
								MAX_Z_DIM_NUM_BLOCKS)
					);

	// Array initialiser notation; cast is put on preprocessor constants as they have small values and will fit in GridDimType (whereas metacell_dims values, of type GridDimType, may not fit in the datatypes automatically assigned to these small constants, and can produce a warning regarding value narrowing)
	GridDimType threads_per_block_dims[Dims::NUM_DIMS] = {
															std::min(metacell_dims[Dims::X_DIM_IND],
																		static_cast<GridDimType>(MAX_X_DIM_THREADS_PER_BLOCK)),
															std::min(metacell_dims[Dims::Y_DIM_IND],
																		static_cast<GridDimType>(MAX_Y_DIM_THREADS_PER_BLOCK)),
															std::min(metacell_dims[Dims::Z_DIM_IND],
																		static_cast<GridDimType>(MAX_Z_DIM_THREADS_PER_BLOCK))
														};
	

	dim3 threads_per_block(threads_per_block_dims[Dims::X_DIM_IND],
								threads_per_block_dims[Dims::Y_DIM_IND],
								threads_per_block_dims[Dims::Z_DIM_IND]
							);

	GridDimType warps_per_block = 1;
	for (int i = 0; i < Dims::NUM_DIMS; i++)
		warps_per_block *= threads_per_block_dims[i];
	warps_per_block = warps_per_block / warp_size + (warps_per_block % warp_size == 0 ? 0 : 1);

	// Shared memory requirement calculation; two arrays: one array for per-warp minimal vertex values, one array for per-warp maximal vertex values, each of length warps_per_block
	if (warps_per_block > 1)
	{
		formMetacellTagsGlobal<true><<<num_blocks, threads_per_block, 2 * warps_per_block * sizeof(T)>>>
					(
						vertex_arr_d, metacell_arr_d, warps_per_block,
						pt_grid_dims[Dims::X_DIM_IND], pt_grid_dims[Dims::Y_DIM_IND], pt_grid_dims[Dims::Z_DIM_IND],
						metacell_grid_dims[Dims::X_DIM_IND], metacell_grid_dims[Dims::Y_DIM_IND], metacell_grid_dims[Dims::Z_DIM_IND],
						metacell_dims[Dims::X_DIM_IND], metacell_dims[Dims::Y_DIM_IND], metacell_dims[Dims::Z_DIM_IND]
					);
	}
	else
	{
		formMetacellTagsGlobal<false><<<num_blocks, threads_per_block>>>
					(
						vertex_arr_d, metacell_arr_d, warps_per_block,
						pt_grid_dims[Dims::X_DIM_IND], pt_grid_dims[Dims::Y_DIM_IND], pt_grid_dims[Dims::Z_DIM_IND],
						metacell_grid_dims[Dims::X_DIM_IND], metacell_grid_dims[Dims::Y_DIM_IND], metacell_grid_dims[Dims::Z_DIM_IND],
						metacell_dims[Dims::X_DIM_IND], metacell_dims[Dims::Y_DIM_IND], metacell_dims[Dims::Z_DIM_IND]
					);
	}

	return metacell_arr_d;
};

// Returns number of voxels in reference num_voxels
template <class PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
PointStruct *formVoxelTags(T *const vertex_arr_d, GridDimType pt_grid_dims[Dims::NUM_DIMS],
							GridDimType metacell_dims[Dims::NUM_DIMS], size_t &num_voxels,
							const int dev_ind, const int num_devs)
{
	// Total number of voxels is \prod_{i=1}^Dims::NUM_DIMS (pt_grid_dims[i] - 1)
	// The -1 addend arises because if one surjectively assigns to each metacell the vertex on its volume that has the smallest indices in each dimension, the edges of the point grid where indices are largest in a given direction will have no metacells, as there are no further points to which to interpolate or draw such a metacell
	num_voxels = 1;
	for (int i = 0; i < Dims::NUM_DIMS; i++)
	{
		num_voxels *= pt_grid_dims[i] - 1;
#ifdef DEBUG
		std::cout << "Number of voxels in dimension " << i << ": " << pt_grid_dims[i] - 1 << '\n';
#endif
	}

#ifdef DEBUG
	std::cout << "Number of voxels: " << num_voxels << '\n';
#endif

	// On-device memory allocations for voxel array
	PointStruct *voxel_arr_d;

	gpuErrorCheck(cudaMalloc(&voxel_arr_d, num_voxels * sizeof(PointStruct)),
					"Error in allocating voxel tag storage array on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	// Set grid size to be equal to number of voxels, unless this exceeds the GPU's capabilities, as determined by its compute capability-associated technical specifications
	// Use of decltype to allow for appropriate instantiation of template function std::min (as implicit casting does not take place among its parameters); preprocessor constants here are either sufficiently large to hold GridDimType or at least to avoid type narrowing warnings
	dim3 num_blocks(std::min(static_cast<decltype(MAX_X_DIM_NUM_BLOCKS)>(pt_grid_dims[Dims::X_DIM_IND]),
								MAX_X_DIM_NUM_BLOCKS),
					std::min(static_cast<decltype(MAX_Y_DIM_NUM_BLOCKS)>(pt_grid_dims[Dims::Y_DIM_IND]),
								MAX_Y_DIM_NUM_BLOCKS),
					std::min(static_cast<decltype(MAX_Z_DIM_NUM_BLOCKS)>(pt_grid_dims[Dims::Z_DIM_IND]),
								MAX_Z_DIM_NUM_BLOCKS)
					);

	// Array initialiser notation; cast is put on preprocessor constants as they have small values and will fit in GridDimType (whereas metacell_dims values, of type GridDimType, may not fit in the datatypes automatically assigned to these small constants, and can produce a warning regarding value narrowing)
	// Despite metacells being largely irrelevant to this function, use metacell_dims to determine dimensionality of a thread block for ease of use and consistency of testing in comparison to formMetacellTags()
	GridDimType threads_per_block_dims[Dims::NUM_DIMS] = {
															std::min(metacell_dims[Dims::X_DIM_IND],
																		static_cast<GridDimType>(MAX_X_DIM_THREADS_PER_BLOCK)),
															std::min(metacell_dims[Dims::Y_DIM_IND],
																		static_cast<GridDimType>(MAX_Y_DIM_THREADS_PER_BLOCK)),
															std::min(metacell_dims[Dims::Z_DIM_IND],
																		static_cast<GridDimType>(MAX_Z_DIM_THREADS_PER_BLOCK))
														};

	dim3 threads_per_block(threads_per_block_dims[Dims::X_DIM_IND],
								threads_per_block_dims[Dims::Y_DIM_IND],
								threads_per_block_dims[Dims::Z_DIM_IND]
							);

	formVoxelTagsGlobal<<<num_blocks, threads_per_block>>>
				(
					vertex_arr_d, voxel_arr_d,
					pt_grid_dims[Dims::X_DIM_IND], pt_grid_dims[Dims::Y_DIM_IND], pt_grid_dims[Dims::Z_DIM_IND]
				);

	return voxel_arr_d;
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

#include "isosurface-data-processing.tu"

#endif
