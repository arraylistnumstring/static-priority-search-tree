// Shared memory must be at least as large (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Non-member functions can only use at most one template clause
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void twoSidedLeftSearchTreeArrGlobal(T *const tree_arr_d,
												const size_t full_tree_num_elem_slots,
												const size_t full_tree_size_num_Ts,
												const size_t num_elems,
												RetType *const res_arr_d,
												const T max_dim1_val, const T min_dim2_val
											)
{
	/*
		By C++ specification 13.7.5 (Templates > Template declarations > Friends), point 9, "a friend function template with a constraint that depends on a template parameter from an enclosing template [is] a definition[...and] does not declare the same[...]function template as a declaration in any other scope."

		Moreover, a requires clause is part of a function's signature, such that the presence or lack of a requires clause changes the function which is called or referenced, so all requires clauses must be removed from the corresponding friend function.

		Finally, attempting to use a requires clause for differently named template parameters (e.g. when declaring a friend template) causes a multiple-overload compilation failure, even when such parameters would have evaluated to equivalent dependencies and function signatures.

		Hence, the static_assert here takes the place of the equivalent requires clause.
	*/
	static_assert(std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
					>::value,
					"RetType is not of type PointStructTemplate<T, IDType, num_IDs>, nor of type IDType");

	cooperative_groups::thread_block curr_block = cooperative_groups::this_thread_block();

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	// For interwarp reductions
	unsigned long long &block_level_start_ind = *reinterpret_cast<unsigned long long *>(s);
	unsigned long long *warp_level_num_elems_arr = reinterpret_cast<unsigned long long *>(s) + 1;
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(warp_level_num_elems_arr + blockDim.x / warpSize + (blockDim.x % warpSize == 0 ? 0 : 1));
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char *>(search_inds_arr + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 in each block start by being inactive
	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = 0;
	else
		search_inds_arr[threadIdx.x] = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;

	// For twoSidedLeftSearchTreeArrGlobal(), all threads start with their search code set to LEFT_SEARCH
	search_codes_arr[threadIdx.x] = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH;

	// Calculate number of slots in this thread block's assigned tree
	size_t tree_num_elem_slots;
	if (num_elems % full_tree_num_elem_slots == 0)		// All trees are full trees
		tree_num_elem_slots = full_tree_num_elem_slots;
	else	// All trees except last are full trees
	{
		if (blockIdx.x < gridDim.x - 1)
			tree_num_elem_slots = full_tree_num_elem_slots;
		// Last tree has (num_elems % full_tree_num_elem_slots) elements
		else
		{
			const size_t last_tree_num_elems = num_elems % full_tree_num_elem_slots;

			tree_num_elem_slots = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcNumElemSlotsPerTree(last_tree_num_elems);
		}
	}

	// Note: would only need to have one thread block do multiple trees when the number of trees exceeds 2^31 - 1, i.e. the maximum number of blocks permitted in a grid
	T *const tree_root_d = tree_arr_d + blockIdx.x * full_tree_size_num_Ts;

	curr_block.sync();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];
	unsigned char search_code = search_codes_arr[threadIdx.x];
	unsigned target_thread_offset = minPowerOf2GreaterThan(threadIdx.x);

	while (cont_iter)
	{
		T curr_node_dim1_val;
		T curr_node_dim2_val;
		T curr_node_med_dim1_val;
		unsigned char curr_node_bitcode;

		bool active_node = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next curr_block.sync() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically
		if (search_ind != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim1_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getDim1ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getDim2ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getMedDim1ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(tree_root_d, tree_num_elem_slots)[search_ind];
		}

		if (min_dim2_val > curr_node_dim2_val)
		{
			search_inds_arr[threadIdx.x] = search_ind
					= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
		}
		else	// Thread stays active with respect to this node
		{
			// Check if current node satisfies query and should be reported
			active_node = curr_node_dim1_val <= max_dim1_val;
		}

		// Report step
		// Intrawarp and interwarp prefix sum
		const unsigned long long block_level_offset
				= calcAllocReportIndOffset<unsigned long long>(curr_block, active_node ? 1 : 0,
																warp_level_num_elems_arr,
																block_level_start_ind);

		if (active_node)
		{
			// Intrawarp prefix sum: each thread here has one active node to report
			//const unsigned long long res_ind_to_access = calcAllocReportIndOffset<unsigned long long>(1);

			if constexpr (std::is_same<RetType, IDType>::value)
			{
				res_arr_d[block_level_start_ind + block_level_offset]
						= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			}
			else
			{
				res_arr_d[block_level_start_ind + block_level_offset].dim1_val = curr_node_dim1_val;
				res_arr_d[block_level_start_ind + block_level_offset].dim2_val = curr_node_dim2_val;
				if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
					res_arr_d[block_level_start_ind + block_level_offset].id
							= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			}
		}

		// Check if a currently active thread will become inactive because current node has no children
		if (search_ind != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND
				&& !GPUTreeNode::hasChildren(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind
				= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
		}


		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE ~> active writes (with ~> denoting a state change that is externally triggered, i.e. triggered by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		curr_block.sync();

		// active threads -> active threads
		// INACTIVE threads -> active threads (activation by active threads only)
		if (search_ind != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH)	// Currently a search-type query
			{
				StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(
																curr_node_med_dim1_val <= max_dim1_val,
																curr_node_bitcode,
																tree_root_d, tree_num_elem_slots,
																res_arr_d, min_dim2_val,
																target_thread_offset,
																search_ind, search_inds_arr,
																search_code, search_codes_arr
															);
			}
			else	// Already a report all-type query
			{
				StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::doReportAllNodesDelegation(
																curr_node_bitcode,
																tree_root_d, tree_num_elem_slots,
																res_arr_d, min_dim2_val,
																target_thread_offset,
																search_ind, search_inds_arr,
																search_codes_arr
															);
			}
		}

		/*
			No curr_block.sync() call necessary here:
				detInactivity() has no false positives (early, incorrect loop exits):
					detInactivity() does not exit even without a curr_block.sync() call here
						<=> There are active threads after previous curr_block.sync() call
						<=> There are some active threads when entering the active -> active, INACTIVE ~> active phase
								(By the nature of the active -> active, INACTIVE ~> active phase, such threads will still be active upon reaching this line)
						<=> There are active threads in the search, i.e. search is ongoing (at least in this block, which is the only source of nodes to search that can be communicated to threads in this block anyway)
						<=> Loop should continue
						<=> cont_iter == true
				detInactivity() has no false negatives (additional, unnecessary loops):
					Unnecessary loops can only occur when threads are incorrectly polled as active
						<=> detInactivity() runs in some threads before all potential active -> inactive transition computations have completed
					This is impossible, as active -> inactive transition computations occur before the previous curr_block.sync() call and will thus always complete for all threads before any thread calls detInactivity() in this iteration
		*/

		StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::detInactivity(
													target_thread_offset, search_ind,
													search_inds_arr, cont_iter,
													&search_code, search_codes_arr
												);

		// No curr_block.sync() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop
	}
	// End cont_iter loop
}
	}
}
