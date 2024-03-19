// Utilises dynamic parallelism
template <typename T>
__global__ void populateTree (T *const root_d, const size_t num_elem_slots, PointStructGPU<T> *const pt_arr_d, size_t *const dim1_val_ind_arr_d, size_t *dim2_val_ind_arr_d, size_t *dim2_val_ind_arr_secondary_d, const size_t val_ind_arr_start_ind, const size_t num_elems, const size_t target_node_start_ind)
{
	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

	extern __shared__ size_t s[];
	size_t *subelems_start_inds_arr = s;
	size_t *num_subelems_arr = s + blockDim.x;
	size_t *target_node_inds_arr = s + (blockDim.x << 1);
	// Initialise shared memory
	subelems_start_inds_arr[threadIdx.x] = val_ind_arr_start_ind;
	// All threads except for thread 0 start by being inactive
	num_subelems_arr[threadIdx.x] = 0;
	if (threadIdx.x == 0)
		num_subelems_arr[threadIdx.x] = num_elems;
	target_node_inds_arr[threadIdx.x] = target_node_start_ind;

	__syncthreads();


	size_t left_subarr_num_elems;
	size_t right_subarr_start_ind;
	size_t right_subarr_num_elems;

	// At any given level of the tree, each thread creates one node; as depth h of a tree has 2^h nodes, the number of active threads at level h is 2^h
	for (size_t nodes_per_level = 1; nodes_per_level < blockDim.x; nodes_per_level <<= 1)
	{
		// Only active threads process a node
		if (threadIdx.x < nodes_per_level && num_subelems_arr[threadIdx.x] > 0)
		{
			// Find index in dim1_val_ind_arr_d of PointStructGPU with maximal dim2_val 
			long long array_search_res_ind = StaticPSTGPU<T>::binarySearch(pt_arr_d, dim1_val_ind_arr_d,
																			pt_arr_d[dim2_val_ind_arr_d[subelems_start_inds_arr[threadIdx.x]]],
																			subelems_start_inds_arr[threadIdx.x],
																			num_subelems_arr[threadIdx.x]);

			if (array_search_res_ind == -1)
				// Something has gone very wrong; exit
				return;

			// Note: potential sign conversion issue when computer memory becomes of size 2^64
			const size_t max_dim2_val_dim1_array_ind = array_search_res_ind;

			StaticPSTGPU<T>::constructNode(root_d, num_elem_slots,
											pt_arr_d, target_node_inds_arr[threadIdx.x], num_elems,
											dim1_val_ind_arr_d, dim2_val_ind_arr_d,
												dim2_val_ind_arr_secondary_d,
												max_dim2_val_dim1_array_ind,
											subelems_start_inds_arr, num_subelems_arr,
											left_subarr_num_elems, right_subarr_start_ind,
											right_subarr_num_elems);

			// Update information for next iteration; as memory accesses are coalesced no matter the relative order as long as they are from the same source location, (and nodes are consecutive except possibly at the leaf levels), pick an inactive thread to instantiate the right child
			if (threadIdx.x + nodes_per_level < blockDim.x)
			{
				subelems_start_inds_arr[threadIdx.x + nodes_per_level] = right_subarr_start_ind;
				num_subelems_arr[threadIdx.x + nodes_per_level] = right_subarr_num_elems;
				target_node_inds_arr[threadIdx.x + nodes_per_level] =
					StaticPSTGPU<T>::TreeNode::getRightChild(target_node_inds_arr[threadIdx.x]);
			}
			// Dynamic parallelism; use cudaStreamFireAndForget to allow children grids to be independent of each other
			else
			{
				populateTree<<<1, blockDim.x, blockDim.x * sizeof(size_t) * 3, cudaStreamFireAndForget>>>
					(root_d, num_elem_slots, pt_arr_d, dim1_val_ind_arr_d,
						dim2_val_ind_arr_d, dim2_val_ind_arr_secondary_d,
						right_subarr_start_ind, right_subarr_num_elems,
						StaticPSTGPU<T>::TreeNode::getRightChild(target_node_inds_arr[threadIdx.x]));
			}

			num_subelems_arr[threadIdx.x] = left_subarr_num_elems;
			target_node_inds_arr[threadIdx.x] =
				StaticPSTGPU<T>::TreeNode::getLeftChild(target_node_inds_arr[threadIdx.x]);
		}

		// Every thread must swap its primary and secondary dim2_val_ind_arr pointers in order to have the correct subordering of indices at a given node
		size_t *temp = dim2_val_ind_arr_d;
		dim2_val_ind_arr_d = dim2_val_ind_arr_secondary_d;
		dim2_val_ind_arr_secondary_d = temp;

		__syncthreads();	// Synchronise before starting the next iteration
	}

	// Each thread is now active and independent of each other
	while (num_subelems_arr[threadIdx.x] > 0)
	{
		// Find index in dim1_val_ind_arr_d of PointStructGPU with maximal dim2_val 
		long long array_search_res_ind = StaticPSTGPU<T>::binarySearch(pt_arr_d, dim1_val_ind_arr_d,
																		pt_arr_d[dim2_val_ind_arr_d[subelems_start_inds_arr[threadIdx.x]]],
																		subelems_start_inds_arr[threadIdx.x],
																		num_subelems_arr[threadIdx.x]);

		if (array_search_res_ind == -1)
			// Something has gone very wrong; exit
			return;

		// Note: potential sign conversion issue when computer memory becomes of size 2^64
		const size_t max_dim2_val_dim1_array_ind = array_search_res_ind;

		StaticPSTGPU<T>::constructNode(root_d, num_elem_slots,
										pt_arr_d, target_node_inds_arr[threadIdx.x], num_elems,
										dim1_val_ind_arr_d, dim2_val_ind_arr_d,
											dim2_val_ind_arr_secondary_d,
											max_dim2_val_dim1_array_ind,
										subelems_start_inds_arr, num_subelems_arr,
										left_subarr_num_elems, right_subarr_start_ind,
										right_subarr_num_elems);

		// Update information for next iteration
		populateTree<<<1, blockDim.x, blockDim.x * sizeof(size_t) * 3, cudaStreamFireAndForget>>>
			(root_d, num_elem_slots, pt_arr_d, dim1_val_ind_arr_d,
				dim2_val_ind_arr_d, dim2_val_ind_arr_secondary_d,
				right_subarr_start_ind, right_subarr_num_elems,
				StaticPSTGPU<T>::TreeNode::getRightChild(target_node_inds_arr[threadIdx.x]));

		num_subelems_arr[threadIdx.x] = left_subarr_num_elems;
		target_node_inds_arr[threadIdx.x] =
			StaticPSTGPU<T>::TreeNode::getLeftChild(target_node_inds_arr[threadIdx.x]);

		size_t *temp = dim2_val_ind_arr_d;
		dim2_val_ind_arr_d = dim2_val_ind_arr_secondary_d;
		dim2_val_ind_arr_secondary_d = temp;
	}
}
