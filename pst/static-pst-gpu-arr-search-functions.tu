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
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(s);
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char *>(search_inds_arr + blockDim.x);

	// Place declarations here so that can be initialised when shared memory is initialised
	long long search_ind;
	unsigned char search_code;

	// Initialise shared memory
	// All threads except for thread 0 in each block start by being inactive
	search_inds_arr[threadIdx.x] = search_ind = 0;
	// For twoSidedLeftSearchTreeArrGlobal(), thread 0 has its search code set to LEFT_SEARCH, while all others have their search code set to REPORT_ABOVE (since splits will only ever result in REPORT_ABOVEs being delegated)
	search_codes_arr[threadIdx.x] = search_code = threadIdx.x == 0 ?
													StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH
														: StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED;

	// Take advantage of potential speed-ups associated with doing local variable updates while waiting for shared memory to be initialised
	cooperative_groups::thread_block::arrival_token shared_mem_init_arrival_token = curr_block.barrier_arrive();

	// Calculate number of slots in this thread block's assigned tree
	const size_t tree_num_elem_slots = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcCurrTreeNumElemSlots(num_elems, full_tree_num_elem_slots);

	// Note: would only need to have one thread block do multiple trees when the number of trees exceeds 2^31 - 1, i.e. the maximum number of blocks permitted in a grid
	T *const tree_root_d = tree_arr_d + blockIdx.x * full_tree_size_num_Ts;

	unsigned target_thread_offset = minPowerOf2GreaterThan(threadIdx.x);

	// Must synchronise before processing to ensure data is properly set
	curr_block.barrier_wait(std::move(shared_mem_init_arrival_token));


	while (cont_iter)
	{
		T curr_node_dim1_val;
		T curr_node_dim2_val;
		T curr_node_med_dim1_val;
		unsigned char curr_node_bitcode;

		bool active_node = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next curr_block.sync() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically
		if (search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED
				&& search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED
			)
		{
			curr_node_dim1_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getDim1ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getDim2ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getMedDim1ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(tree_root_d, tree_num_elem_slots)[search_ind];
		}

		if (min_dim2_val > curr_node_dim2_val)
		{
			search_codes_arr[threadIdx.x] = search_code
					= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED;
		}
		else	// Thread stays active with respect to this node
		{
			// Check if current node satisfies query and should be reported
			active_node = curr_node_dim1_val <= max_dim1_val;
		}

		// Report step
		if (active_node)
		{
			// Intrawarp prefix sum: each thread here has one active node to report
			const unsigned long long res_ind_to_access = calcAllocReportIndOffset<unsigned long long>(1);

			if constexpr (std::is_same<RetType, IDType>::value)
			{
				res_arr_d[res_ind_to_access]
						= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			}
			else
			{
				res_arr_d[res_ind_to_access].dim1_val = curr_node_dim1_val;
				res_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
					res_arr_d[res_ind_to_access].id
							= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			}
		}

		// Check if a currently active thread will become inactive because current node has no children
		if (search_ind != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED
				&& search_ind != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED
				&& !GPUTreeNode::hasChildren(curr_node_bitcode))
		{
			search_codes_arr[threadIdx.x] = search_code
				= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED;
		}

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE ~> active writes (with ~> denoting a state change that is externally triggered, i.e. triggered by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		curr_block.sync();


		// active threads -> active threads; each active thread whose search splits sends a "wake-up" message to an INACTIVE thread's shared memory slot, for that (currently INACTIVE) thread to later read and become active
		if (search_ind != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED
				&& search_ind != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED)
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
				StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::doReportAboveDelegation(
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
			curr_block.sync() call necessary here:
				If a delegator thread delegates to the current thread, completes its search and deactivates itself while the current thread has yet to completely execute detInactivity() (i.e. by checking its shared memory slot before the delegator thread has delegated to it, then not progressing further in the code before the delegator thread exits), the current thread may incorrectly fail to get its newly delegated node, run the loop-exiting search instead, see that all potential delegators have exited, and prematurely exit the loop without searching its own assigned subtree

			curr_block.sync() calls delimit the delegation code on both sides because any set of commands that occupy the same curr_block.sync()-delimited chunk can be executed concurrently (by different threads) with no guarantee as to their ordering. Thus, in a loop, because the end loops back to the beginning (until the iteration condition is broken), having only one curr_block.sync() call is equivalent to the loop being comprised of only one curr_block.sync()-delimited chunk
		*/
		curr_block.sync();


		// INACTIVE threads -> active threads (by reading their shared memory slots if activated by an already-active thread); or INACTIVE threads -> exit loop (if all possible threads that could activate this thread have already become inactive)
		if (search_ind == StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED)
			StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::detInactivity(search_ind,
																					search_inds_arr,
																					cont_iter,
																					search_code,
																					search_codes_arr
																				);

		// No curr_block.sync() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop
	}
	// End cont_iter loop
}
