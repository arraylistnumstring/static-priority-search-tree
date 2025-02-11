template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__global__ void populateTrees(T *const tree_arr_d, const size_t full_tree_num_elem_slots,
								const size_t full_tree_size_num_Ts,
								PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
								size_t *const dim1_val_ind_arr_d,
								size_t *dim2_val_ind_arr_d,
								size_t *dim2_val_ind_arr_secondary_d,
								const size_t num_elems)
{
	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	size_t *subelems_start_inds_arr = reinterpret_cast<size_t *>(s);
	size_t *num_subelems_arr = reinterpret_cast<size_t *>(s) + blockDim.x;
	size_t *target_node_inds_arr = reinterpret_cast<size_t *>(s) + (blockDim.x << 1);
	// Initialise shared memory
	subelems_start_inds_arr[threadIdx.x] = 0;
	// All threads except for thread 0 start by being inactive
	num_subelems_arr[threadIdx.x] = 0;
	if (threadIdx.x == 0)
		num_subelems_arr[threadIdx.x] = num_elems;
	target_node_inds_arr[threadIdx.x] = 0;

	// Note: would only need to have one thread block do multiple trees if the number of trees exceeds 2^31 - 1, i.e. the maximum number of blocks permitted in a grid
	T *const tree_root_d = tree_arr_d + blockIdx.x * full_tree_size_num_Ts;

	__syncthreads();

	// At any given level of the tree, each thread creates one node; as depth h of a tree has 2^h nodes, the number of active threads at level h is 2^h
	// num_subelems_arr[threadIdx.x] > 0 condition not written here, as almost all threads except for thread 0 start out with the value num_subelems_arr[threadIdx.x] == 0
#pragma unroll
	for (size_t nodes_per_level = 1; nodes_per_level <= leastPowerOf2(blockDim.x);
			nodes_per_level <<= 1)
	{
		// Only active threads process a node
		if (threadIdx.x < nodes_per_level && num_subelems_arr[threadIdx.x] > 0)
		{
			// Location of these declarations and presence of pragma unroll don't affect register usage (with CUDA version 12.2), so place closer to semantically relevant location
			size_t left_subarr_num_elems;
			size_t right_subarr_start_ind;
			size_t right_subarr_num_elems;

			// Find index in dim1_val_ind_arr_d of PointStruct with maximal dim2_val 
			long long array_search_res_ind = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::binarySearch(pt_arr_d, dim1_val_ind_arr_d,
												pt_arr_d[dim2_val_ind_arr_d[subelems_start_inds_arr[threadIdx.x]]],
												subelems_start_inds_arr[threadIdx.x],
												num_subelems_arr[threadIdx.x]);

			if (array_search_res_ind == -1)
				// Something has gone very wrong; exit
				return;

			// Note: potential sign conversion issue when computer memory becomes of size 2^64
			const size_t max_dim2_val_dim1_array_ind = array_search_res_ind;
		}
	}
}
