template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__global__ void populateTrees(T *const tree_arr_d, const size_t full_tree_num_elem_slots,
								const size_t full_tree_size_num_Ts,
								PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
								const size_t num_elems,
								size_t *const dim1_val_ind_arr_d,
								size_t *dim2_val_ind_arr_d,
								size_t *dim2_val_ind_arr_secondary_d)
{
	cooperative_groups::thread_block curr_block = cooperative_groups::this_thread_block();

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	size_t *subelems_start_inds_arr = reinterpret_cast<size_t *>(s);
	size_t *num_subelems_arr = reinterpret_cast<size_t *>(s) + blockDim.x;
	size_t *target_tree_node_inds_arr = reinterpret_cast<size_t *>(s) + (blockDim.x << 1);
	// Initialise shared memory
	subelems_start_inds_arr[threadIdx.x] = blockIdx.x * full_tree_num_elem_slots;
	target_tree_node_inds_arr[threadIdx.x] = 0;

	// Calculate number of slots in this thread block's assigned tree
	size_t tree_num_elem_slots;
	if (num_elems % full_tree_num_elem_slots == 0)		// All trees are full trees
	{
		tree_num_elem_slots = full_tree_num_elem_slots;

		// All threads except for thread 0 in each block start by being inactive
		if (threadIdx.x == 0)
			num_subelems_arr[threadIdx.x] = tree_num_elem_slots;
		else
			num_subelems_arr[threadIdx.x] = 0;
	}
	else	// All trees except last are full trees
	{
		// All blocks except last populate full trees
		if (blockIdx.x < gridDim.x - 1)
		{
			tree_num_elem_slots = full_tree_num_elem_slots;

			// All threads except for thread 0 start by being inactive
			if (threadIdx.x == 0)
				num_subelems_arr[threadIdx.x] = tree_num_elem_slots;
			else
				num_subelems_arr[threadIdx.x] = 0;
		}
		// Last tree has (num_elems % full_tree_num_elem_slots) elements
		else
		{
			const size_t last_tree_num_elems = num_elems % full_tree_num_elem_slots;

			tree_num_elem_slots = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcNumElemSlotsPerTree(last_tree_num_elems);

			// All threads except for thread 0 start by being inactive
			if (threadIdx.x == 0)
				num_subelems_arr[threadIdx.x] = last_tree_num_elems;
			else
				num_subelems_arr[threadIdx.x] = 0;
		}
	}

	// Note: would only need to have one thread block do multiple trees when the number of trees exceeds 2^31 - 1, i.e. the maximum number of blocks permitted in a grid
	T *const tree_root_d = tree_arr_d + blockIdx.x * full_tree_size_num_Ts;

	curr_block.sync();


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

			StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::constructNode(
											tree_root_d, tree_num_elem_slots,
											pt_arr_d, target_tree_node_inds_arr[threadIdx.x],
											dim1_val_ind_arr_d, dim2_val_ind_arr_d,
												dim2_val_ind_arr_secondary_d,
												max_dim2_val_dim1_array_ind,
											subelems_start_inds_arr[threadIdx.x],
											num_subelems_arr[threadIdx.x],
											left_subarr_num_elems, right_subarr_start_ind,
											right_subarr_num_elems
										);

			// Update information for next iteration; as memory accesses are coalesced no matter the relative order as long as they are from the same source location, (and nodes are consecutive except possibly at the leaf levels), pick an inactive thread to instantiate the right child
			
			// Save current target_tree_node_ind and save left child information first in order to allow for reuse of variables left_subarr_num_elems, right_subarr_start_ind, right_subarr_num_elems in right child construction, decreasing overall register usage
			const size_t curr_target_tree_node_ind = target_tree_node_inds_arr[threadIdx.x];

			num_subelems_arr[threadIdx.x] = left_subarr_num_elems;
			target_tree_node_inds_arr[threadIdx.x] = GPUTreeNode::getLeftChild(curr_target_tree_node_ind);

			// If there exist inactive threads in the block, assign the right child to an inactive thread and the left child to oneself
			if (threadIdx.x + nodes_per_level < blockDim.x)
			{
				subelems_start_inds_arr[threadIdx.x + nodes_per_level] = right_subarr_start_ind;
				num_subelems_arr[threadIdx.x + nodes_per_level] = right_subarr_num_elems;
				target_tree_node_inds_arr[threadIdx.x + nodes_per_level] =
					GPUTreeNode::getRightChild(curr_target_tree_node_ind);
			}
			// Because of how elements have been allocated to this tree, this means that the next level is the last level; in this case, if there are no more threads available to construct the right child, do so (it will not have any children, so if it exists, it will be the last node to handle in its subtree)
			else if (right_subarr_num_elems > 0)
			{
				// left_subarr_num_elems, right_subarr_start_ind, right_subarr_num_elems are not used further in this iteration, so can reuse them for construction of the right child leaf
				StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::constructNode(
												tree_root_d, tree_num_elem_slots,
												pt_arr_d,
												GPUTreeNode::getRightChild(curr_target_tree_node_ind),
												dim1_val_ind_arr_d,
													// Must switch primary and secondary dim2 arrays at next level
													dim2_val_ind_arr_secondary_d, dim2_val_ind_arr_d,
													max_dim2_val_dim1_array_ind,
												// const pass-by-value
												right_subarr_start_ind, right_subarr_num_elems,
												// pass by reference
												left_subarr_num_elems, right_subarr_start_ind,
												right_subarr_num_elems
											);

				if (left_subarr_num_elems != 0 || right_subarr_num_elems != 0)
					// Something has gone very wrong; exit
					return;
			}
		}

		// Every thread must swap its primary and secondary dim2_val_ind_arr pointers in order to have the correct subordering of indices at a given node
		size_t *temp = dim2_val_ind_arr_d;
		dim2_val_ind_arr_d = dim2_val_ind_arr_secondary_d;
		dim2_val_ind_arr_secondary_d = temp;

		curr_block.sync();	// Synchronise before starting the next iteration
	}
}
