// For use of std::move(); CUDA automatically gives it __host__ __device__ qualifiers, unless explicitly specified against during compilation (CUDA Programming Guide 14.5.22.3)
#include <utility>

#include "calc-alloc-report-ind-offset.h"
#include "gpu-tree-node.h"


// Utilises dynamic parallelism
// Shared memory must be at least as large as (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Correctness only guaranteed for grids with one active block
// Non-member functions can only use at most one template clause
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void threeSidedSearchGlobal(T *const root_d, const size_t num_elem_slots,
										const long long start_node_ind,
										RetType *const res_arr_d,
										const T min_dim1_val, const T max_dim1_val,
										const T min_dim2_val)
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

	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

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
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = search_ind = threadIdx.x == 0 ? start_node_ind
													: StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
	// For threeSidedSearchGlobal(), each thread starts out with their code set to THREE_SEARCH
	search_codes_arr[threadIdx.x] = search_code = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::THREE_SEARCH;

	// Take advantage of potential speed-ups associated with doing local variable updates while waiting for shared memory to be initialised
	cooperative_groups::thread_block::arrival_token shared_mem_init_arrival_token = curr_block.barrier_arrive();

	bool cont_iter = true;	// Loop-continuing flag

	// Must synchronise before processing to ensure data is properly set
	curr_block.barrier_wait(std::move(shared_mem_init_arrival_token));


	while (cont_iter)
	{
		// Set arrival token here, which, due to the nature of the loop, is effectively an arrival token for the end of detNextIterState(); as detNextIterState() and this section of code are all writes to one's own state, there is no race condition between these two sections
		cooperative_groups::thread_block::arrival_token det_next_iter_state_arrival_token
				= curr_block.barrier_arrive();

		T curr_node_dim1_val;
		T curr_node_dim2_val;
		T curr_node_med_dim1_val;
		unsigned char curr_node_bitcode;

		bool active_node = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next curr_block.sync() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim1_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getMedDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];

			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
			}
			else	// Thread stays active with respect to this node
			{
				// Check if current node satisfies query and should be reported
				active_node = min_dim1_val <= curr_node_dim1_val
								&& curr_node_dim1_val <= max_dim1_val;
			}
		}


		// Report step

		/*
			Note that though interwarp reductions are possible and correct if interwarp calcAllocReportIndOffset() is called here (along with adding the associated shared memory slots, shared memory declarations and substituting res_arr_d[res_arr_ind_to_access] with res_arr_d[block_level_res_start_ind + thread_offset_in_block]), interwarp reductions are only occasionally faster (by about ~0.03 ms) and generally slower (by about ~0.1 ms) than intrawarp-only reductions
		const unsigned long long thread_offset_in_block
				= calcAllocReportIndOffset<unsigned long long>(curr_block, active_node ? 1 : 0,
																warp_level_num_elems_arr,
																block_level_res_start_ind);
		*/

		if (active_node)
		{
			// Intrawarp prefix sum: each thread here has one active node to report
			const unsigned long long res_ind_to_access = calcAllocReportIndOffset<unsigned long long>(1);

			if constexpr (std::is_same<RetType, IDType>::value)
			{
				res_arr_d[res_ind_to_access]
					= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
			else
			{
				res_arr_d[res_ind_to_access].dim1_val = curr_node_dim1_val;
				res_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				// As IDs are only accessed if the node is to be reported and if IDs exist, don't waste a register on it (and avoid compilation failures from attempting to instantiate a potential void variable)
				if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
					res_arr_d[res_ind_to_access].id
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
		}

		// Check if a currently active node will become inactive because current node: a) has no children; or b) the subtree satisfying the search range has no children
		// Entails an update to search_ind, so must come after report step, which uses search_ind to retrieve IDs to report
		// Deactivation must occur on this side of the following syncthreads() call to avoid race conditions
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND
				&& (!GPUTreeNode::hasChildren(curr_node_bitcode)
					|| (max_dim1_val < curr_node_med_dim1_val
							&& !GPUTreeNode::hasLeftChild(curr_node_bitcode))
					|| (curr_node_med_dim1_val < min_dim1_val
							&& !GPUTreeNode::hasRightChild(curr_node_bitcode))
					)
			)
		{
			search_inds_arr[threadIdx.x] = search_ind
				= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
		}

		// In order for any delegation code to run correctly, it can only be run when there are no threads attempting to exit the loop (in case of scheduling causing concurrent interleaving between a delegator's delegation commands and a delegatee's check of its shared memory slot)
		curr_block.barrier_wait(std::move(det_next_iter_state_arrival_token));


		// active threads -> active threads; each active thread whose search splits sends a "wake-up" message to an INACTIVE thread's shared memory slot, for that (currently INACTIVE) thread to later read and become active
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::THREE_SEARCH)	// Currently a three-sided query
			{
				// Do 3-sided search delegation
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::do3SidedSearchDelegation(
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d,
															min_dim1_val, max_dim1_val,
															curr_node_med_dim1_val, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr
														);
			}
			else if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH)
			{
				// Do left search delegation
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(
															curr_node_med_dim1_val <= max_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr
														);
			}
			else if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::RIGHT_SEARCH)
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doRightSearchDelegation(
															curr_node_med_dim1_val >= min_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr
														);
			}
			else	// search_code == REPORT_ABOVE
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAboveDelegation(
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_codes_arr
														);
			}
		}

		/*
			curr_block.sync() call necessary here:
				If a delegator thread delegates to the current thread, completes its search and deactivates itself while the current thread has yet to completely execute detNextIterState() (i.e. by checking its shared memory slot before the delegator thread has delegated to it, then not progressing further in the code before the delegator thread exits), the current thread may incorrectly fail to get its newly delegated node, run the loop-exiting search instead and, in the (low-probability, but not impossible) case that all other threads have exited, prematurely exit the loop without searching its own assigned subtree

			curr_block.sync() calls delimit the delegation code on both sides because any set of commands that occupy the same curr_block.sync()-delimited chunk can be executed concurrently (by different threads) with no guarantee as to their ordering. Thus, in a loop, because the end loops back to the beginning (until the iteration condition is broken), having only one curr_block.sync() call is equivalent to the loop being comprised of only one curr_block.sync()-delimited chunk
		*/
		curr_block.sync();


		if (search_ind == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
			StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detNextIterState(search_ind,
																					search_inds_arr,
																					cont_iter,
																					&search_code,
																					search_codes_arr
																				);
	}
	// End cont_iter loop
}

// Shared memory must be at least as large as (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Correctness only guaranteed for grids with one active block
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void twoSidedLeftSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const long long start_node_ind,
											RetType *const res_arr_d,
											const T max_dim1_val, const T min_dim2_val)
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

	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

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
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = search_ind = threadIdx.x == 0 ? start_node_ind
													: StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
	// For twoSidedLeftSearchGlobal(), thread 0 has its search code set to LEFT_SEARCH, while all others have their search code set to REPORT_ABOVE (since splits will only ever result in REPORT_ABOVEs being delegated)
	search_codes_arr[threadIdx.x] = search_code = threadIdx.x == 0 ?
														StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH
															: StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::REPORT_ABOVE;

	// Take advantage of potential speed-ups associated with doing local variable updates while waiting for shared memory to be initialised
	cooperative_groups::thread_block::arrival_token shared_mem_init_arrival_token = curr_block.barrier_arrive();

	bool cont_iter = true;	// Loop-continuing flag

	// Must synchronise before processing to ensure data is properly set
	curr_block.barrier_wait(std::move(shared_mem_init_arrival_token));


	while (cont_iter)
	{
		// Set arrival token here, which, due to the nature of the loop, is effectively an arrival token for the end of detNextIterState(); as detNextIterState() and this section of code are all writes to one's own state, there is no race condition between these two sections
		cooperative_groups::thread_block::arrival_token det_next_iter_state_arrival_token
				= curr_block.barrier_arrive();

		T curr_node_dim1_val;
		T curr_node_dim2_val;
		T curr_node_med_dim1_val;
		unsigned char curr_node_bitcode;

		bool active_node = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next curr_block.sync() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim1_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getMedDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];

			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
			}
			else	// Thread stays active with respect to this node
			{
				// Check if current node satisfies query and should be reported
				active_node = curr_node_dim1_val <= max_dim1_val;
			}
		}


		// Report step

		/*
			Note that though interwarp reductions are possible and correct if interwarp calcAllocReportIndOffset() is called here (along with adding the associated shared memory slots, shared memory declarations and substituting res_arr_d[res_arr_ind_to_access] with res_arr_d[block_level_res_start_ind + thread_offset_in_block]), interwarp reductions are only occasionally faster (by about ~0.03 ms) and generally slower (by about ~0.1 ms) than intrawarp-only reductions
		const unsigned long long thread_offset_in_block
				= calcAllocReportIndOffset<unsigned long long>(curr_block, active_node ? 1 : 0,
																warp_level_num_elems_arr,
																block_level_res_start_ind);
		*/

		if (active_node)
		{
			// Intrawarp prefix sum: each thread here has one active node to report
			const unsigned long long res_ind_to_access = calcAllocReportIndOffset<unsigned long long>(1);

			if constexpr (std::is_same<RetType, IDType>::value)
			{
				res_arr_d[res_ind_to_access]
					= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
			else
			{
				res_arr_d[res_ind_to_access].dim1_val = curr_node_dim1_val;
				res_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
					res_arr_d[res_ind_to_access].id
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
		}

		// Check if a currently active thread will become inactive because current node has no children
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND
				&& !GPUTreeNode::hasChildren(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind
				= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
		}

		// In order for any delegation code to run correctly, it can only be run when there are no threads attempting to exit the loop (in case of scheduling causing concurrent interleaving between a delegator's delegation commands and a delegatee's check of its shared memory slot)
		curr_block.barrier_wait(std::move(det_next_iter_state_arrival_token));


		// active threads -> active threads; each active thread whose search splits sends a "wake-up" message to an INACTIVE thread's shared memory slot, for that (currently INACTIVE) thread to later read and become active
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH)	// Currently a search-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(
															curr_node_med_dim1_val <= max_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr
														);
			}
			else	// Already a report all-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAboveDelegation(
																curr_node_bitcode,
																root_d, num_elem_slots,
																res_arr_d, min_dim2_val,
																search_ind, search_inds_arr,
																search_codes_arr
															);
			}
		}

		/*
			curr_block.sync() call necessary here:
				If a delegator thread delegates to the current thread, completes its search and deactivates itself while the current thread has yet to completely execute detNextIterState() (i.e. by checking its shared memory slot before the delegator thread has delegated to it, then not progressing further in the code before the delegator thread exits), the current thread may incorrectly fail to get its newly delegated node, run the loop-exiting search instead and, in the (low-probability, but not impossible) case that all other threads have exited, prematurely exit the loop without searching its own assigned subtree

			curr_block.sync() calls delimit the delegation code on both sides because any set of commands that occupy the same curr_block.sync()-delimited chunk can be executed concurrently (by different threads) with no guarantee as to their ordering. Thus, in a loop, because the end loops back to the beginning (until the iteration condition is broken), having only one curr_block.sync() call is equivalent to the loop being comprised of only one curr_block.sync()-delimited chunk
		*/
		curr_block.sync();


		if (search_ind == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
			StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detNextIterState(search_ind,
																					search_inds_arr,
																					cont_iter,
																					&search_code,
																					search_codes_arr
																				);
	}
	// End cont_iter loop
}

// Shared memory must be at least as large as (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Correctness only guaranteed for grids with one active block
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void twoSidedRightSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const long long start_node_ind,
											RetType *const res_arr_d,
											const T min_dim1_val, const T min_dim2_val)
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

	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

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
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = search_ind = threadIdx.x == 0 ? start_node_ind
													: StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
	// For twoSidedRightSearchGlobal(), thread 0 has its search code set to RIGHT_SEARCH, while all others have their search code set to REPORT_ABOVE (since splits will only ever result in REPORT_ABOVEs being delegated)
	search_codes_arr[threadIdx.x] = search_code = threadIdx.x == 0 ?
													StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::RIGHT_SEARCH
														: StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::REPORT_ABOVE;

	// Take advantage of potential speed-ups associated with doing local variable updates while waiting for shared memory to be initialised
	cooperative_groups::thread_block::arrival_token shared_mem_init_arrival_token = curr_block.barrier_arrive();

	bool cont_iter = true;	// Loop-continuing flag

	// Must synchronise before processing to ensure data is properly set
	curr_block.barrier_wait(std::move(shared_mem_init_arrival_token));


	while (cont_iter)
	{
		// Set arrival token here, which, due to the nature of the loop, is effectively an arrival token for the end of detNextIterState(); as detNextIterState() and this section of code are all writes to one's own state, there is no race condition between these two sections
		cooperative_groups::thread_block::arrival_token det_next_iter_state_arrival_token
				= curr_block.barrier_arrive();

		T curr_node_dim1_val;
		T curr_node_dim2_val;
		T curr_node_med_dim1_val;
		unsigned char curr_node_bitcode;
		bool active_node = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next curr_block.sync() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim1_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getMedDim1ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];

			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
			}
			else	// Thread stays active with respect to this node
			{
				// Check if current node satisfies query and should be reported
				active_node = curr_node_dim1_val >= min_dim1_val;
			}
		}


		// Report step

		/*
			Note that though interwarp reductions are possible and correct if interwarp calcAllocReportIndOffset() is called here (along with adding the associated shared memory slots, shared memory declarations and substituting res_arr_d[res_arr_ind_to_access] with res_arr_d[block_level_res_start_ind + thread_offset_in_block]), interwarp reductions are only occasionally faster (by about ~0.03 ms) and generally slower (by about ~0.1 ms) than intrawarp-only reductions
		const unsigned long long thread_offset_in_block
				= calcAllocReportIndOffset<unsigned long long>(curr_block, active_node ? 1 : 0,
																warp_level_num_elems_arr,
																block_level_res_start_ind);
		*/

		if (active_node)
		{
			// Intrawarp prefix sum: each thread here has one active node to report
			const unsigned long long res_ind_to_access = calcAllocReportIndOffset<unsigned long long>(1);

			if constexpr (std::is_same<RetType, IDType>::value)
			{
				res_arr_d[res_ind_to_access]
					= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
			else
			{
				res_arr_d[res_ind_to_access].dim1_val = curr_node_dim1_val;
				res_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
					res_arr_d[res_ind_to_access].id
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
		}

		// Check if a currently active thread becomes inactive because current node has no children
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND
				&& !GPUTreeNode::hasChildren(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind
				= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
		}

		// In order for any delegation code to run correctly, it can only be run when there are no threads attempting to exit the loop (in case of scheduling causing concurrent interleaving between a delegator's delegation commands and a delegatee's check of its shared memory slot)
		curr_block.barrier_wait(std::move(det_next_iter_state_arrival_token));


		// active threads -> active threads; each active thread whose search splits sends a "wake-up" message to an INACTIVE thread's shared memory slot, for that (currently INACTIVE) thread to later read and become active
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::RIGHT_SEARCH)	// Currently a search-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doRightSearchDelegation(
															curr_node_med_dim1_val >= min_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr
														);
			}
			else	// Already a report all-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAboveDelegation(
																curr_node_bitcode,
																root_d, num_elem_slots,
																res_arr_d, min_dim2_val,
																search_ind, search_inds_arr,
																search_codes_arr
															);
			}
		}

		/*
			curr_block.sync() call necessary here:
				If a delegator thread delegates to the current thread, completes its search and deactivates itself while the current thread has yet to completely execute detNextIterState() (i.e. by checking its shared memory slot before the delegator thread has delegated to it, then not progressing further in the code before the delegator thread exits), the current thread may incorrectly fail to get its newly delegated node, run the loop-exiting search instead and, in the (low-probability, but not impossible) case that all other threads have exited, prematurely exit the loop without searching its own assigned subtree

			curr_block.sync() calls delimit the delegation code on both sides because any set of commands that occupy the same curr_block.sync()-delimited chunk can be executed concurrently (by different threads) with no guarantee as to their ordering. Thus, in a loop, because the end loops back to the beginning (until the iteration condition is broken), having only one curr_block.sync() call is equivalent to the loop being comprised of only one curr_block.sync()-delimited chunk
		*/
		curr_block.sync();


		if (search_ind == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
			StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detNextIterState(search_ind,
																					search_inds_arr,
																					cont_iter,
																					&search_code,
																					search_codes_arr
																				);
	}
	// End cont_iter loop
}

// Shared memory must be at least as large as (number of threads) * sizeof(long long)
// Correctness only guaranteed for grids with one active block
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void reportAboveGlobal(T *const root_d, const size_t num_elem_slots,
										const long long start_node_ind,
										RetType *const res_arr_d,
										const T min_dim2_val)
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

	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

	cooperative_groups::thread_block curr_block = cooperative_groups::this_thread_block();

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(s);

	// Place declaration here so that can be initialised when shared memory is initialised
	long long search_ind;

	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	search_inds_arr[threadIdx.x] = search_ind = threadIdx.x == 0 ? start_node_ind
												: StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;

	// Take advantage of potential speed-ups associated with doing local variable updates while waiting for shared memory to be initialised
	cooperative_groups::thread_block::arrival_token shared_mem_init_arrival_token = curr_block.barrier_arrive();

	bool cont_iter = true;	// Loop-continuing flag

	// Must synchronise before processing to ensure data is properly set
	curr_block.barrier_wait(std::move(shared_mem_init_arrival_token));


	while (cont_iter)
	{
		// Set arrival token here, which, due to the nature of the loop, is effectively an arrival token for the end of detNextIterState(); as detNextIterState() and this section of code are all writes to one's own state, there is no race condition between these two sections
		cooperative_groups::thread_block::arrival_token det_next_iter_state_arrival_token
				= curr_block.barrier_arrive();

		// curr_node_dim1_val will only be accessed at most once (during reporting if RetType == PointStructs) so no need to create an automatic variable for it
		T curr_node_dim2_val;
		unsigned char curr_node_bitcode;
		bool active_node = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next curr_block.sync() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			curr_node_dim2_val = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim2ValsRoot(root_d, num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(root_d, num_elem_slots)[search_ind];
			
			if (min_dim2_val > curr_node_dim2_val)
			{
				search_inds_arr[threadIdx.x] = search_ind
					= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
			}
			// Thread stays active with respect to this node
			else	// min_dim2_val <= curr_node_dim2_val; report node
				active_node = true;
		}


		// Report step

		/*
			Note that though interwarp reductions are possible and correct if interwarp calcAllocReportIndOffset() is called here (along with adding the associated shared memory slots, shared memory declarations and substituting res_arr_d[res_arr_ind_to_access] with res_arr_d[block_level_res_start_ind + thread_offset_in_block]), interwarp reductions are only occasionally faster (by about ~0.03 ms) and generally slower (by about ~0.1 ms) than intrawarp-only reductions
		const unsigned long long thread_offset_in_block
				= calcAllocReportIndOffset<unsigned long long>(curr_block, active_node ? 1 : 0,
																warp_level_num_elems_arr,
																block_level_res_start_ind);
		*/

		if (active_node)
		{
			// Intrawarp prefix sum: each thread here has one active node to report
			const unsigned long long res_ind_to_access = calcAllocReportIndOffset<unsigned long long>(1);

			if constexpr (std::is_same<RetType, IDType>::value)
			{
				res_arr_d[res_ind_to_access]
					= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
			else
			{
				res_arr_d[res_ind_to_access].dim1_val
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getDim1ValsRoot(root_d, num_elem_slots)[search_ind];
				res_arr_d[res_ind_to_access].dim2_val = curr_node_dim2_val;
				if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
					res_arr_d[res_ind_to_access].id
						= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::getIDsRoot(root_d, num_elem_slots)[search_ind];
			}
		}

		// Check if thread becomes inactive because current node has no children
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND
				&& !GPUTreeNode::hasChildren(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind
				= StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
		}

		// In order for any delegation code to run correctly, it can only be run when there are no threads attempting to exit the loop (in case of scheduling causing concurrent interleaving between a delegator's delegation commands and a delegatee's check of its shared memory slot)
		curr_block.barrier_wait(std::move(det_next_iter_state_arrival_token));


		// active threads -> active threads; each active thread whose search splits sends a "wake-up" message to an INACTIVE thread's shared memory slot, for that (currently INACTIVE) thread to later read and become active
		// If thread remains active, it must have at least one child
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAboveDelegation(
																curr_node_bitcode,
																root_d, num_elem_slots,
																res_arr_d, min_dim2_val,
																search_ind, search_inds_arr
															);
		}

		/*
			curr_block.sync() call necessary here:
				If a delegator thread delegates to the current thread, completes its search and deactivates itself while the current thread has yet to completely execute detNextIterState() (i.e. by checking its shared memory slot before the delegator thread has delegated to it, then not progressing further in the code before the delegator thread exits), the current thread may incorrectly fail to get its newly delegated node, run the loop-exiting search instead and, in the (low-probability, but not impossible) case that all other threads have exited, prematurely exit the loop without searching its own assigned subtree

			curr_block.sync() calls delimit the delegation code on both sides because any set of commands that occupy the same curr_block.sync()-delimited chunk can be executed concurrently (by different threads) with no guarantee as to their ordering. Thus, in a loop, because the end loops back to the beginning (until the iteration condition is broken), having only one curr_block.sync() call is equivalent to the loop being comprised of only one curr_block.sync()-delimited chunk
		*/
		curr_block.sync();


		if (search_ind == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
			StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detNextIterState(search_ind,
																					search_inds_arr,
																					cont_iter
																				);
	}
	// End cont_iter loop
}
