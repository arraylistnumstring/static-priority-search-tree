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

				// Generate mask for threads active during intrawarp phase; all threads in warp run this (or else are exited, i.e. simply not running any code at all)
				// Call to __ballot_sync() is necessary to determine the thread in warp with largest ID that is still active
				// As of time of writing (compute capability 9.0), __ballot_sync() returns an unsigned int
				const auto intrawarp_mask = __ballot_sync(0xffffffff, valid_voxel);

				if (valid_voxel)
				{
					getVoxelMinMax(vertex_arr_d, min_vert_val, max_vert_val,
										base_vertex_coord_x, base_vertex_coord_y, base_vertex_coord_z,
										pt_grid_dims_x, pt_grid_dims_y, pt_grid_dims_z);

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


					// Intrawarp reduction for metacell min-max val determination
#ifdef DEBUG
					printf("About to begin metacell intrawarp reduce\n");
#endif

					// CUDA-supplied __reduce_*_sync() is only defined for types unsigned and int, and isn't even found for some reason when compiling, so use user-defined warpReduce() instead
					min_vert_val = warpReduce(intrawarp_mask, min_vert_val, min_op);
					max_vert_val = warpReduce(intrawarp_mask, max_vert_val, max_op);

#ifdef DEBUG
					printf("Completed metacell intrawarp reduce\n");
#endif
				}

				// Interwarp reduction for metacell min-max val determination
				if constexpr (interwarp_reduce)
				{
					// Note: nvcc does not use more or fewer registers when lin_thread_ID_in_block is replaced with linThreadIDInBlock; however, generally speaking, saving the result to a const variable like this is more performant than repeating a function call
					const auto lin_thread_ID_in_block = linThreadIDInBlock();

					// First thread in each warp writes result to shared memory
					if (lin_thread_ID_in_block % warpSize == 0)
					{
						warp_level_min_vert[lin_thread_ID_in_block / warpSize] = min_vert_val;
						warp_level_max_vert[lin_thread_ID_in_block / warpSize] = max_vert_val;
					}

					// Warp-level info must be ready to use at the block level
					__syncthreads();

					// Only one warp should be active for speed and correctness
					if (lin_thread_ID_in_block / warpSize == 0)
					{
#pragma unroll
						for (GridDimType l = 0; l < warps_per_block; l += warpSize)
						{
							const auto interwarp_mask = __ballot_sync(0xffffffff,
																		l + lin_thread_ID_in_block < warps_per_block);

							// Inter-warp condition
							if (l + lin_thread_ID_in_block < warps_per_block)
							{
								// Get per-warp minimum and maximum vertex values
								min_vert_val = warp_level_min_vert[lin_thread_ID_in_block / warpSize];
								max_vert_val = warp_level_max_vert[lin_thread_ID_in_block / warpSize];

								min_vert_val = warpReduce(interwarp_mask, min_vert_val, min_op);
								max_vert_val = warpReduce(interwarp_mask, min_vert_val, max_op);
							}
						}
					}
				}

				// All threads in first warp have the correct overall result for the metacell; single thread in block writes result to global memory array
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
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
__forceinline__ __device__ void getVoxelMinMax(T *const vertex_arr_d, T &min, T &max,
													const GridDimType base_vertex_coord_x,
													const GridDimType base_vertex_coord_y,
													const GridDimType base_vertex_coord_z,
													const GridDimType pt_grid_dims_x,
													const GridDimType pt_grid_dims_y,
													const GridDimType pt_grid_dims_z
												)
{
	// Each thread accesses the vertex of its voxel with the lowest indices in each dimension and uses this scalar value as the initial value with which future values are compared
	max = min = vertex_arr_d[lineariseID(base_vertex_coord_x, base_vertex_coord_y, base_vertex_coord_z, pt_grid_dims_x, pt_grid_dims_y)];

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
				min = min <= curr_vert_val ? min : curr_vert_val;
				max = max >= curr_vert_val ? max : curr_vert_val;
			}
		}
	}
}
