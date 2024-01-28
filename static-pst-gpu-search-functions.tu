// Utilises dynamic parallelism
template <typename T>
__global__ void threeSidedSearchGlobal(T *const root_d, const size_t num_elem_slots, const size_t start_node_ind, PointStructGPU<T> *const res_pt_arr_d, const T min_dim1_val, const T max_dim1_val, const T min_dim2_val)
{
	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

	extern __shared__ long long search_shared_mem[];
	// Node indices for each thread to search
	long long *search_inds_arr = search_shared_mem;
	// bool is a 1-byte datatype
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char*>(search_shared_mem + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
	// For threeSidedSearchGlobal, each thread starts out with their code set to THREE_SEARCH
	search_codes_arr[threadIdx.x] = StaticPSTGPU<T>::SearchCodes::THREE_SEARCH;

	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];
	unsigned char search_code = search_codes_arr[threadIdx.x];

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;

	while (cont_iter)
	{
		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim1_val = StaticPSTGPU<T>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPU<T>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPU<T>::getMedDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];

			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
			}
			else	// Thread stays active with respect to this node
			{
				// Check if current node satisfies query and should be reported
				if (min_dim1_val <= curr_node_dim1_val
						&& curr_node_dim1_val <= max_dim1_val)
				{
					unsigned long long res_ind_to_access = atomicAdd(&res_arr_ind_d, 1);
					res_pt_arr_d[res_ind_to_access].dim1_val = curr_node_dim1_val;
					res_pt_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				}

				// If node has no children or the subtree satisfying the search range has no children, the thread becomes inactive; inactivity must occur on this side of the following syncthreads() call to avoid race conditions
				if (!StaticPSTGPU<T>::TreeNode::hasChildren(curr_node_bitcode)
						|| (max_dim1_val < curr_node_med_dim1_val
								&& !StaticPSTGPU<T>::TreeNode::hasLeftChild(curr_node_bitcode))
						|| (curr_node_med_dim1_val < min_dim1_val
								&& !StaticPSTGPU<T>::TreeNode::hasRightChild(curr_node_bitcode)))
				{
					search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
				}
			}
		}

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T>::SearchCodes::THREE_SEARCH)	// Currently a three-sided query
			{
				// Splitting of query is only possible if the current node has two children and min_dim1_val <= curr_node_med_dim1_val <= max_dim1_val; the equality on max_dim1_val is for the edge case where a median point may be duplicated, with one copy going to the left subtree and the other to the right subtree
				if (min_dim1_val <= curr_node_med_dim1_val
						&& curr_node_med_dim1_val <= max_dim1_val)
				{
					// Query splits over median and node has two children; split into 2 two-sided queries
					if (StaticPSTGPU<T>::TreeNode::hasLeftChild(curr_node_bitcode)
							&& StaticPSTGPU<T>::TreeNode::hasRightChild(curr_node_bitcode))
					{
						// Delegate work of searching right subtree to another thread and/or block
						StaticPSTGPU<T>::splitLeftSearchWork(root_d, num_elem_slots,
																StaticPSTGPU<T>::TreeNode::getRightChild(search_ind),
																res_pt_arr_d,
																max_dim1_val, min_dim2_val,
																search_inds_arr, search_codes_arr);

						// Prepare to search left subtree with a two-sided right search in the next iteration
						search_inds_arr[threadIdx.x] = search_ind
							= StaticPSTGPU<T>::TreeNode::getLeftChild(search_ind);
						search_codes_arr[threadIdx.x] = search_code
							= StaticPSTGPU<T>::SearchCodes::RIGHT_SEARCH;
					}
					// No right child, so perform a two-sided right query on the left child
					else if (StaticPSTGPU<T>::TreeNode::hasLeftChild(curr_node_bitcode))
					{
						search_inds_arr[threadIdx.x] = search_ind
							= StaticPSTGPU<T>::TreeNode::getLeftChild(search_ind);
						search_codes_arr[threadIdx.x] = search_code
							= StaticPSTGPU<T>::SearchCodes::RIGHT_SEARCH;
					}
					// No left child, so perform a two-sided left query on the right child
					else
					{
						search_inds_arr[threadIdx.x] = search_ind
							= StaticPSTGPU<T>::TreeNode::getRightChild(search_ind);
						search_codes_arr[threadIdx.x] = search_code
							= StaticPSTGPU<T>::SearchCodes::LEFT_SEARCH;
					}
				}
				// Perform three-sided search on left child
				else if (max_dim1_val < curr_node_med_dim1_val
							&& StaticPSTGPU<T>::TreeNode::hasLeftChild(curr_node_bitcode))
				{
					// Search code is already a THREE_SEARCH
					search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::TreeNode::getLeftChild(search_ind);
				}
				// Perform three-sided search on right child
				// Only remaining possibility, as all others mean the thread is inactive:
				//		curr_node_med_dim1_val < min_dim1_val && StaticPSTGPU<T>::TreeNode::hasRightChild(curr_node_bitcode)
				else
				{
					// Search code is already a THREE_SEARCH
					search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::TreeNode::getRightChild(search_ind);
				}
			}
			else if (search_code == StaticPSTGPU<T>::SearchCodes::LEFT_SEARCH)
			{
				// Do left search delegation
				StaticPSTGPU<T>::doLeftSearchDelegation(curr_node_med_dim1_val <= max_dim1_val,
										curr_node_bitcode,
										root_d, num_elem_slots,
										res_pt_arr_d, min_dim2_val,
										search_ind, search_inds_arr,
										search_code, search_codes_arr);
			}
			else if (search_code == StaticPSTGPU<T>::SearchCodes::RIGHT_SEARCH)
			{
				StaticPSTGPU<T>::doRightSearchDelegation(curr_node_med_dim1_val >= min_dim1_val,
											curr_node_bitcode, root_d, num_elem_slots,
											res_pt_arr_d, min_dim2_val,
											search_ind, search_inds_arr,
											search_code, search_codes_arr);
			}
			else	// search_code == REPORT_ALL
			{
				StaticPSTGPU<T>::doReportAllNodesDelegation(curr_node_bitcode, root_d, num_elem_slots,
											res_pt_arr_d, min_dim2_val,
											search_ind, search_inds_arr,
											search_codes_arr);
			}
		}


		__syncthreads();

		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T>::detInactivity(search_ind, search_inds_arr, cont_iter, &search_code, search_codes_arr);
	}
	// End cont_iter loop
}

template <typename T>
__global__ void twoSidedLeftSearchGlobal(T *const root_d, const size_t num_elem_slots, const size_t start_node_ind, PointStructGPU<T> *const res_pt_arr_d, const T max_dim1_val, const T min_dim2_val)
{
	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

	extern __shared__ long long search_shared_mem[];
	// Node indices for each thread to search
	long long *search_inds_arr = search_shared_mem;
	// bool is a 1-byte datatype
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char*>(search_shared_mem + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
	// For twoSidedLeftSearchGlobal, all threads start with their search code set to LEFT_SEARCH 
	search_codes_arr[threadIdx.x] = StaticPSTGPU<T>::SearchCodes::LEFT_SEARCH;

	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];
	unsigned char search_code = search_codes_arr[threadIdx.x];

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;

	while (cont_iter)
	{
		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim1_val = StaticPSTGPU<T>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPU<T>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPU<T>::getMedDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];

			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
			}
			else	// Thread stays active with respect to this node
			{
				// Check if current node satisfies query and should be reported
				if (curr_node_dim1_val <= max_dim1_val)
				{
					unsigned long long res_ind_to_access = atomicAdd(&res_arr_ind_d, 1);
					res_pt_arr_d[res_ind_to_access].dim1_val = curr_node_dim1_val;
					res_pt_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				}

				// Check if thread becomes inactive because current node has no children
				if (!StaticPSTGPU<T>::TreeNode::hasChildren(curr_node_bitcode))
				{
					search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
				}
			}
		}

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T>::SearchCodes::LEFT_SEARCH)	// Currently a search-type query
			{
				StaticPSTGPU<T>::doLeftSearchDelegation(curr_node_med_dim1_val <= max_dim1_val,
										curr_node_bitcode,
										root_d, num_elem_slots,
										res_pt_arr_d, min_dim2_val,
										search_ind, search_inds_arr,
										search_code, search_codes_arr);
			}
			else	// Already a report all-type query
			{
				StaticPSTGPU<T>::doReportAllNodesDelegation(curr_node_bitcode, root_d, num_elem_slots,
											res_pt_arr_d, min_dim2_val,
											search_ind, search_inds_arr,
											search_codes_arr);
			}
		}

		__syncthreads();

		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T>::detInactivity(search_ind, search_inds_arr, cont_iter, &search_code, search_codes_arr);
	}
	// End cont_iter loop
}

template <typename T>
__global__ void twoSidedRightSearchGlobal(T *const root_d, const size_t num_elem_slots, const size_t start_node_ind, PointStructGPU<T> *const res_pt_arr_d, const T min_dim1_val, const T min_dim2_val)
{
	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

	extern __shared__ long long search_shared_mem[];
	// Node indices for each thread to search
	long long *search_inds_arr = search_shared_mem;
	// bool is a 1-byte datatype
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char*>(search_shared_mem + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
	// For twoSidedRightSearchGlobal, all threads start with their search code set to RIGHT_SEARCH
	search_codes_arr[threadIdx.x] = StaticPSTGPU<T>::SearchCodes::RIGHT_SEARCH;

	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];
	unsigned char search_code = search_codes_arr[threadIdx.x];

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;

	while (cont_iter)
	{
		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim1_val = StaticPSTGPU<T>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPU<T>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPU<T>::getMedDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];

			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
			}
			else	// Thread stays active with respect to this node
			{
				// Check if current node satisfies query and should be reported
				if (curr_node_dim1_val >= min_dim1_val)
				{
					unsigned long long res_ind_to_access = atomicAdd(&res_arr_ind_d, 1);
					res_pt_arr_d[res_ind_to_access].dim1_val = curr_node_dim1_val;
					res_pt_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				}

				// Check if thread becomes inactive because current node has no children
				if (!StaticPSTGPU<T>::TreeNode::hasChildren(curr_node_bitcode))
				{
					search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
				}
			}
		}

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T>::SearchCodes::RIGHT_SEARCH)	// Currently a search-type query
			{
				StaticPSTGPU<T>::doRightSearchDelegation(curr_node_med_dim1_val >= min_dim1_val,
											curr_node_bitcode, root_d, num_elem_slots,
											res_pt_arr_d, min_dim2_val,
											search_ind, search_inds_arr,
											search_code, search_codes_arr);
			}
			else	// Already a report all-type query
			{
				StaticPSTGPU<T>::doReportAllNodesDelegation(curr_node_bitcode, root_d, num_elem_slots,
											res_pt_arr_d, min_dim2_val,
											search_ind, search_inds_arr,
											search_codes_arr);
			}
		}

		__syncthreads();


		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T>::detInactivity(search_ind, search_inds_arr, cont_iter, &search_code, search_codes_arr);
	}
	// End cont_iter loop
}

template <typename T>
__global__ void reportAllNodesGlobal(T *const root_d, const size_t num_elem_slots, const size_t start_node_ind, PointStructGPU<T> *const res_pt_arr_d, const T min_dim2_val)
{
	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

	extern __shared__ long long search_shared_mem[];
	// Node indices for each thread to search
	long long *search_inds_arr = search_shared_mem;
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;

	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];

	// curr_node_dim1_val will only be accessed once, so no need to create an automatic variable for it
	T curr_node_dim2_val;
	unsigned char curr_node_bitcode;

	while (cont_iter)
	{
		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim2_val = StaticPSTGPU<T>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];
			
			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
					= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
			}
			// Thread stays active with respect to this node
			else	// min_dim2_val <= curr_node_dim2_val; report node
			{
				unsigned long long res_ind_to_access = atomicAdd(&res_arr_ind_d, 1);
				res_pt_arr_d[res_ind_to_access].dim1_val
						= StaticPSTGPU<T>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
				res_pt_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;

				// Check if thread becomes inactive because current node has no children
				if (!StaticPSTGPU<T>::TreeNode::hasChildren(curr_node_bitcode))
				{
					search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::IndexCodes::INACTIVE_IND;
				}
			}
		}

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		// If thread remains active, it must have at least one child
		if (search_ind != StaticPSTGPU<T>::IndexCodes::INACTIVE_IND)
		{
			if (StaticPSTGPU<T>::TreeNode::hasLeftChild(curr_node_bitcode)
					&& StaticPSTGPU<T>::TreeNode::hasRightChild(curr_node_bitcode))
			{
				// Delegate reporting of all nodes in right child to another thread and/or block
				StaticPSTGPU<T>::splitReportAllNodesWork(root_d, num_elem_slots,
															StaticPSTGPU<T>::TreeNode::getRightChild(search_ind),
															res_pt_arr_d, min_dim2_val,
															search_inds_arr);

				// Prepare to report all nodes in the next iteration
				search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T>::TreeNode::getLeftChild(search_ind);
			}
			// Node only has a left child; report all on left child
			else if (StaticPSTGPU<T>::TreeNode::hasLeftChild(curr_node_bitcode))
			{
				search_inds_arr[threadIdx.x] = search_ind
					= StaticPSTGPU<T>::TreeNode::getLeftChild(search_ind);
			}
			// Node only has a right child; report all on right child
			else if (StaticPSTGPU<T>::TreeNode::hasRightChild(curr_node_bitcode))
			{
				search_inds_arr[threadIdx.x] = search_ind
					= StaticPSTGPU<T>::TreeNode::getRightChild(search_ind);
			}
		}

		__syncthreads();


		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T>::detInactivity(search_ind, search_inds_arr, cont_iter);
	}
	// End cont_iter loop
}
