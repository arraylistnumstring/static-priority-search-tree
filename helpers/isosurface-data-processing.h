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


template <class PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
PointStruct *formMetacells(T *const vertex_arr_d, GridDimType pt_grid_dims[Dims::NUM_DIMS],
							GridDimType metacell_dims[Dims::NUM_DIMS], size_t &num_metacells,
							const int dev_ind, const int num_devs, const int warp_size)
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
	PointStruct *metacell_arr_d;

	gpuErrorCheck(cudaMalloc(&metacell_arr_d, num_metacells * sizeof(PointStruct)),
					"Error in allocating metacell storage array on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");

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
		formMetacellsGlobal<true><<<num_blocks, threads_per_block, 2 * warps_per_block * sizeof(T)>>>
					(
						vertex_arr_d, metacell_arr_d, warps_per_block,
						pt_grid_dims[Dims::X_DIM_IND], pt_grid_dims[Dims::Y_DIM_IND], pt_grid_dims[Dims::Z_DIM_IND],
						metacell_grid_dims[Dims::X_DIM_IND], metacell_grid_dims[Dims::Y_DIM_IND], metacell_grid_dims[Dims::Z_DIM_IND],
						metacell_dims[Dims::X_DIM_IND], metacell_dims[Dims::Y_DIM_IND], metacell_dims[Dims::Z_DIM_IND]
					);
	}
	else
	{
		formMetacellsGlobal<false><<<num_blocks, threads_per_block>>>
					(
						vertex_arr_d, metacell_arr_d, warps_per_block,
						pt_grid_dims[Dims::X_DIM_IND], pt_grid_dims[Dims::Y_DIM_IND], pt_grid_dims[Dims::Z_DIM_IND],
						metacell_grid_dims[Dims::X_DIM_IND], metacell_grid_dims[Dims::Y_DIM_IND], metacell_grid_dims[Dims::Z_DIM_IND],
						metacell_dims[Dims::X_DIM_IND], metacell_dims[Dims::Y_DIM_IND], metacell_dims[Dims::Z_DIM_IND]
					);
	}

	return metacell_arr_d;
};

// Must have point grid and metacell grid dimension values available, in case there is a discrepancy between the maximal possible thread block size and actual metacell size; and/or maximal possible thread grid size and actual metacell grid size
// To minimise global memory access time (and because the number of objects passed is relatively small for each set of dimensions), use explicitly passed scalar parameters for each dimension
template <bool interwarp_reduce, typename PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
__global__ void formMetacellsGlobal(T *const vertex_arr_d, PointStruct *const metacell_arr_d,
										const GridDimType warps_per_block,
										GridDimType pt_grid_dims_x, GridDimType pt_grid_dims_y, GridDimType pt_grid_dims_z,
										GridDimType metacell_grid_dims_x, GridDimType metacell_grid_dims_y, GridDimType metacell_grid_dims_z,
										GridDimType metacell_dims_x, GridDimType metacell_dims_y, GridDimType metacell_dims_z
									)
{
	extern __shared__ char s[];
	T *warp_level_min_vert = reinterpret_cast<T *>(s);
	T *warp_level_max_vert = reinterpret_cast<T *>(s) + warps_per_block;

	T min_vert_val, max_vert_val;
	// Repeat over entire voxel grid
	// Data is z-major, then y-major, then x-major (i.e. x-dimension index changes the fastest, followed by y-index, then z-index)
	// Entire block is active if at least one thread is active
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
				bool active_voxel = base_voxel_coord_x < pt_grid_dims_x - 1
										&& base_voxel_coord_y < pt_grid_dims_y - 1
										&& base_voxel_coord_z < pt_grid_dims_z - 1;
				if (active_voxel)
				{
					getVoxelMinMax(vertex_arr_d, min_vert_val, max_vert_val,
										base_voxel_coord_x, base_voxel_coord_y, base_voxel_coord_z,
										pt_grid_dims_x, pt_grid_dims_y);
				}

				// Intrawarp reduction for metacell min-max val determination

				// Generate mask for threads active during intrawarp phase; all threads in warp run this (or else are exited, i.e. simply not running any code at all)
				// Call to __ballot_sync() is necessary to determine the thread in warp with largest ID that is still active
				// As of time of writing (compute capability 9.0), __ballot_sync() returns an unsigned int
				const auto intrawarp_mask = __ballot_sync(0xffffffff, active_voxel);

				// Neither CUDA math library-provided min() and max() functions nor host-only std::min() and std::max() compile when passed as parameters to device functions, so simply use lambdas (that implicitly cast to nvstd::function type when assigned to a variable of that type)
				nvstd::function<T(const T &, const T &)> min_op
						= [](const T &num1, const T &num2) -> T
							{
								return num1 <= num2 ? num1 : num2;
							};

				nvstd::function<T(const T &, const T &)> max_op
						= [](const T &num1, const T &num2) -> T
							{
								return num1 >= num2 ? num1 : num2;
							};

#ifdef DEBUG
				printf("About to begin metacell intrawarp reduce\n");
#endif

				// CUDA-supplied __reduce_*_sync() is only defined for types unsigned and int, and isn't even found for some reason when compiling, so use user-defined warpReduce() instead
				min_vert_val = warpReduce(intrawarp_mask, min_vert_val, min_op);
				max_vert_val = warpReduce(intrawarp_mask, max_vert_val, max_op);

#ifdef DEBUG
				printf("Completed metacell intrawarp reduce\n");
#endif

				// Interwarp reduction for metacell min-max val determination
				if constexpr (interwarp_reduce)
				{
					const GridDimType lin_thread_ID = lineariseID(threadIdx.x, threadIdx.y, threadIdx.z,
																	blockDim.x, blockDim.y);

					// First thread in each warp writes result to shared memory
					if (lin_thread_ID % warpSize == 0)
					{
						warp_level_min_vert[lin_thread_ID / warpSize] = min_vert_val;
						warp_level_max_vert[lin_thread_ID / warpSize] = max_vert_val;
					}

					// Warp-level info must be ready to use at the block level
					__syncthreads();

					// Only one warp should be active for speed and correctness
					if (lin_thread_ID / warpSize == 0)
					{
#pragma unroll
						for (GridDimType l = 0; l < warps_per_block; l += warpSize)
						{
							const auto interwarp_mask = __ballot_sync(0xffffffff,
																		l + lin_thread_ID < warps_per_block);

							// Inter-warp condition
							if (l + lin_thread_ID < warps_per_block)
							{
								// Get per-warp minimum and maximum vertex values
								min_vert_val = warp_level_min_vert[lin_thread_ID / warpSize];
								max_vert_val = warp_level_max_vert[lin_thread_ID / warpSize];

								min_vert_val = warpReduce(interwarp_mask, min_vert_val, min_op);
								max_vert_val = warpReduce(interwarp_mask, min_vert_val, max_op);
							}
						}
					}
				}

				// All threads in first warp have the correct overall result for the metacell; single thread in block writes result to global memory array
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
				{
					// Cast necessary, as an arithemtic operation (even of two types that are both small, e.g. GridDimType = char) effects an up-casting to a datatype at least as large as int, whereas directly supplied variables remain as the previous type, causing the overall template instantiation of lineariseID to fail
					GridDimType metacellID = lineariseID(base_voxel_coord_x / metacell_dims_x,
															base_voxel_coord_y / metacell_dims_y,
															base_voxel_coord_z / metacell_dims_z,
															metacell_grid_dims_x,
															metacell_grid_dims_y
														);
					metacell_arr_d[metacellID].dim1_val = min_vert_val;
					metacell_arr_d[metacellID].dim2_val = max_vert_val;
					if constexpr (HasID<PointStruct>::value)
						metacell_arr_d[metacellID].id = metacellID;
				}

				// No need for further synchronisation, as all consequent operations before the next interwarp reduction are independent and concurrency-safe
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
				T curr_vert = vertex_arr_d[lineariseID(base_voxel_coord_x + i,
															base_voxel_coord_y + j,
															base_voxel_coord_z + k,
															pt_grid_dims_x, pt_grid_dims_y
														)];
				min = min <= curr_vert ? min : curr_vert;
				max = max >= curr_vert ? max : curr_vert;
			}
		}
	}
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
