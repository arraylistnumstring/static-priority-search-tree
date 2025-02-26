#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "linearise-id.h"


// Must have point grid and metacell grid dimension values available, in case there is a discrepancy between the maximal possible thread block size and actual metacell size; and/or maximal possible thread grid size and actual metacell grid size
// To minimise global memory access time (and because the number of objects passed is relatively small for each set of dimensions), use explicitly passed scalar parameters for each dimension
template <bool interwarp_reduce, typename PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
__global__ void formMetacellTagsGlobal(T *const vertex_arr_d, PointStruct *const metacell_arr_d,
										const GridDimType warps_per_block,
										GridDimType pt_grid_dims_x, GridDimType pt_grid_dims_y, GridDimType pt_grid_dims_z,
										GridDimType metacell_grid_dims_x, GridDimType metacell_grid_dims_y, GridDimType metacell_grid_dims_z,
										GridDimType metacell_dims_x, GridDimType metacell_dims_y, GridDimType metacell_dims_z
									)
{
	extern __shared__ char s[];
	T *warp_level_min_vert = reinterpret_cast<T *>(s);
	T *warp_level_max_vert = reinterpret_cast<T *>(s) + warps_per_block;

	const cooperative_groups::thread_block curr_block = cooperative_groups::this_thread_block();

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
				GridDimType base_vertex_coord_x = i + blockIdx.x * blockDim.x + threadIdx.x;
				GridDimType base_vertex_coord_y = j + blockIdx.y * blockDim.y + threadIdx.y;
				GridDimType base_vertex_coord_z = k + blockIdx.z * blockDim.z + threadIdx.z;
				// One voxel is associated with each vertex, with the exception of the last vertices in each dimension (i.e. those vertices with at least one coordinate of value pt_grid_dims_[x-z] - 1)
				bool valid_voxel = base_vertex_coord_x < pt_grid_dims_x - 1
										&& base_vertex_coord_y < pt_grid_dims_y - 1
										&& base_vertex_coord_z < pt_grid_dims_z - 1;

				// Intrawarp reduce
				if (valid_voxel)
				{
					// coalesced_group is a cooperative group composed of all currently active threads in a warp; it allows for library function-based reduce operations that take in a custom operator
					const cooperative_groups::coalesced_group intrawarp_active_threads
							= cooperative_groups::coalesced_threads();

					getVoxelMinMax(vertex_arr_d, min_vert_val, max_vert_val,
										base_vertex_coord_x, base_vertex_coord_y, base_vertex_coord_z,
										pt_grid_dims_x, pt_grid_dims_y, pt_grid_dims_z);

					// Intrawarp reduction for metacell min-max val determination
#ifdef DEBUG
					printf("About to begin metacell intrawarp reduce\n");
#endif

					// Use CUDA math library-provided min() and max() functions, which are overloaded such that all numeric types have their own associated version
					min_vert_val = cooperative_groups::reduce(intrawarp_active_threads, min_vert_val, min);
					max_vert_val = cooperative_groups::reduce(intrawarp_active_threads, max_vert_val, max);

#ifdef DEBUG
					printf("Completed metacell intrawarp reduce\n");
#endif
				}

				// Interwarp reduction for metacell min-max val determination
				if constexpr (interwarp_reduce)
				{
					// First thread in each warp writes result to shared memory
					if (curr_block.thread_rank() % warpSize == 0)
					{
						warp_level_min_vert[curr_block.thread_rank() / warpSize] = min_vert_val;
						warp_level_max_vert[curr_block.thread_rank() / warpSize] = max_vert_val;
					}

					// Warp-level info must be ready to use at the block level
					// Equivalent to __syncthreads(), as well as to calling curr_block.barrier_wait(curr_block.barrier_arrive()) (i.e. having no code between the arrival event and the wait event (the latter of which stops all threads until all threads have passed the arrival event)
					curr_block.sync();

					// Only one warp should be active for speed and correctness
					if (curr_block.thread_rank() / warpSize == 0)
					{
#pragma unroll
						for (GridDimType l = 0; l < warps_per_block; l += warpSize)
						{
							// Inter-warp condition
							if (l + curr_block.thread_rank() < warps_per_block)
							{
								// coalesced_group is a cooperative group composed of all currently active threads in a warp; it allows for library function-based reduce operations that take in a custom operator
								cooperative_groups::coalesced_group interwarp_active_threads
										= cooperative_groups::coalesced_threads();

								// Get per-warp minimum and maximum vertex values
								min_vert_val = warp_level_min_vert[l + curr_block.thread_rank()];
								max_vert_val = warp_level_max_vert[l + curr_block.thread_rank()];

								min_vert_val = cooperative_groups::reduce(interwarp_active_threads, min_vert_val, min);
								max_vert_val = cooperative_groups::reduce(interwarp_active_threads, max_vert_val, max);
							}
						}
					}
				}

				// All threads in first warp have the correct overall result for the metacell; single thread in block writes result to global memory array
				if (curr_block.thread_rank() == 0)
				{
					// Cast necessary, as an arithmetic operation (even of two types that are both small, e.g. GridDimType = char) effects an up-casting to a datatype at least as large as int, whereas directly supplied variables remain as the previous type, causing the overall template instantiation of lineariseID to fail
					GridDimType metacell_ID = lineariseID(base_vertex_coord_x / metacell_dims_x,
															base_vertex_coord_y / metacell_dims_y,
															base_vertex_coord_z / metacell_dims_z,
															metacell_grid_dims_x,
															metacell_grid_dims_y
														);
					metacell_arr_d[metacell_ID].dim1_val = min_vert_val;
					metacell_arr_d[metacell_ID].dim2_val = max_vert_val;
					if constexpr (HasID<PointStruct>::value)
						metacell_arr_d[metacell_ID].id = metacell_ID;
				}

				// No need for further synchronisation, as all consequent operations before the next interwarp reduction are independent and concurrency-safe
			}
		}
	}
}

template <typename PointStruct, typename T, typename GridDimType>
	requires IntSizeOfUAtLeastSizeOfV<GridDimType, int>
__global__ void formVoxelTagsGlobal(T *const vertex_arr_d, PointStruct *const voxel_arr_d,
										GridDimType pt_grid_dims_x, GridDimType pt_grid_dims_y, GridDimType pt_grid_dims_z
									)
{
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
				GridDimType base_vertex_coord_x = i + blockIdx.x * blockDim.x + threadIdx.x;
				GridDimType base_vertex_coord_y = j + blockIdx.y * blockDim.y + threadIdx.y;
				GridDimType base_vertex_coord_z = k + blockIdx.z * blockDim.z + threadIdx.z;
				// One voxel is associated with each vertex, with the exception of the last vertices in each dimension (i.e. those vertices with at least one coordinate of value pt_grid_dims_[x-z] - 1)
				bool valid_voxel = base_vertex_coord_x < pt_grid_dims_x - 1
										&& base_vertex_coord_y < pt_grid_dims_y - 1
										&& base_vertex_coord_z < pt_grid_dims_z - 1;
				if (valid_voxel)
				{
					getVoxelMinMax(vertex_arr_d, min_vert_val, max_vert_val,
										base_vertex_coord_x, base_vertex_coord_y, base_vertex_coord_z,
										pt_grid_dims_x, pt_grid_dims_y, pt_grid_dims_z);

					// Voxel-specific processing to save to a PointStruct
					// Cast necessary, as an arithmetic operation (even of two types that are both small, e.g. GridDimType = char) effects an up-casting to a datatype at least as large as int, whereas directly supplied variables remain as the previous type, causing the overall template instantiation of lineariseID to fail
					GridDimType voxel_ID = lineariseID(base_vertex_coord_x,
														base_vertex_coord_y,
														base_vertex_coord_z,
														pt_grid_dims_x - 1, pt_grid_dims_y - 1
													);
					voxel_arr_d[voxel_ID].dim1_val = min_vert_val;
					voxel_arr_d[voxel_ID].dim2_val = max_vert_val;
					if constexpr (HasID<PointStruct>::value)
						voxel_arr_d[voxel_ID].id = voxel_ID;
				}
			}
		}
	}
}

template <typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
__forceinline__ __device__ void getVoxelMinMax(T *const vertex_arr_d, T &min_val, T &max_val,
													const GridDimType base_vertex_coord_x,
													const GridDimType base_vertex_coord_y,
													const GridDimType base_vertex_coord_z,
													const GridDimType pt_grid_dims_x,
													const GridDimType pt_grid_dims_y,
													const GridDimType pt_grid_dims_z
												)
{
	// Each thread accesses the vertex of its voxel with the lowest indices in each dimension and uses this scalar value as the initial value with which future values are compared
	max_val = min_val = vertex_arr_d[lineariseID(base_vertex_coord_x, base_vertex_coord_y, base_vertex_coord_z, pt_grid_dims_x, pt_grid_dims_y)];

	// Check each vertex of the voxel and get the maximum and minimum values achieved at those vertices
	for (GridDimType k = 0; k < 2; k++)
	{
		for (GridDimType j = 0; j < 2; j++)
		{
			for (GridDimType i = 0; i < 2; i++)
			{
				// Base voxel coordinate is equal to vertex of lowest dimension
				T curr_vert_val = vertex_arr_d[lineariseID(base_vertex_coord_x + i,
															base_vertex_coord_y + j,
															base_vertex_coord_z + k,
															pt_grid_dims_x, pt_grid_dims_y
														)];
				// Use CUDA math library functions
				min_val = min(min_val, curr_vert_val);
				max_val = max(max_val, curr_vert_val);
			}
		}
	}
}
