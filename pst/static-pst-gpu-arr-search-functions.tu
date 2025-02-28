// For use of std::move(); CUDA automatically gives it __host__ __device__ qualifiers, unless explicitly specified against during compilation (CUDA Programming Guide 14.5.22.3)
#include <utility>

#include "calc-alloc-report-ind-offset.h"
#include "gpu-tree-node.h"


// Shared memory must be at least as large (number of threads) * (sizeof(size_t) + sizeof(unsigned char))
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
	size_t *search_inds_arr = reinterpret_cast<size_t *>(s);
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char *>(search_inds_arr + blockDim.x);

	// Place declarations here so that can be initialised when shared memory is initialised
	size_t search_ind;
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

	// Due to possessing enough threads in each block (and therefore for each tree, up to the leaf level), delegate search splits like in subtree construction: with thread offsets that are successive powers of 2 (but only increasing the offset when a delegation actually occurs)
	unsigned target_thread_offset = minPowerOf2GreaterThan(threadIdx.x);

	// Must synchronise before processing to ensure data is properly set
	curr_block.barrier_wait(std::move(shared_mem_init_arrival_token));


	// By design, threads that have been DEACTIVATED are never re-activated
	while (search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED)
	{
		// Set arrival token here, which, due to the nature of the loop, is effectively an arrival token for the end of detNextIterState(); as detNextIterState() and this section of code are all writes to one's own state, there is no race condition between these two sections
		cooperative_groups::thread_block::arrival_token det_next_iter_state_arrival_token
				= curr_block.barrier_arrive();

		T curr_node_dim1_val;
		T curr_node_dim2_val;
		T curr_node_med_dim1_val;
		unsigned char curr_node_bitcode;

		bool active_node = false;

		// active threads -> DEACTIVATED (if current node goes below the dim2_val threshold or has no children)
		// active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically
		if (search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED
				&& search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED
			)
		{
			curr_node_dim1_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getDim1ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_dim2_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getDim2ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_med_dim1_val = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getMedDim1ValsRoot(tree_root_d, tree_num_elem_slots)[search_ind];
			curr_node_bitcode = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(tree_root_d, tree_num_elem_slots)[search_ind];

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
		}


		// Report step

		// Interwarp prefix sum is infeasible here, as block-level synchronisation cannot filter out threads by activity level and, while the prefix sum itself would be correct among all active threads, the atomic increment of the global space-allocating index variable res_arr_ind_d would need to be done by a determinstically designated thread. Doing so would require said thread to poll all other threads for their activity levels, and would defeat the purpose of a more lightweight deactivation design that only needs to poll the chain of threads that would potentially activate the current thread (O(lg(threadIdx.x)) operations)

		if (active_node)
		{
			// Intrawarp prefix sum: each thread here has one active node to report; works correctly no matter how many threads are not participating
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
		if (search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED
				&& search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED
				&& !GPUTreeNode::hasChildren(curr_node_bitcode))
		{
			search_codes_arr[threadIdx.x] = search_code
				= StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED;
		}

		// DEACTIVATED threads never reactivate, so no race conditions exist regarding writing to their search_codes_arr shared memory slot

		// In order for any delegation code to run correctly, it can only be run when there are no threads attempting to become DEACTIVATED and exit the loop (in case of scheduling causing concurrent interleaving between a delegator's delegation commands and a delegatee's check of its shared memory slot)
		curr_block.barrier_wait(std::move(det_next_iter_state_arrival_token));


		// State transition: active -> active; each active thread whose search splits sends an "activation" message to an UNACTIVATED thread's shared memory slot, for that (currently UNACTIVATED) thread to later read and become active
		if (search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED
				&& search_code != StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::DEACTIVATED)
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
				If a delegator thread delegates to the current thread, completes its search and deactivates itself while the current thread has yet to completely execute detNextIterState() (i.e. by checking its shared memory slot before the delegator thread has delegated to it, then not progressing further in the code before the delegator thread exits), the current thread may incorrectly fail to get its newly delegated node, run the loop-exiting search instead and, if all other threads in its delegation chain have exited, prematurely exit the loop without searching its own assigned subtree

			curr_block.sync() calls delimit the delegation code on both sides because any set of commands that occupy the same curr_block.sync()-delimited chunk can be executed concurrently (by different threads) with no guarantee as to their ordering. Thus, in a loop, because the end loops back to the beginning (until the iteration condition is broken), having only one curr_block.sync() call is equivalent to the loop being comprised of only one curr_block.sync()-delimited chunk
		*/
		curr_block.sync();


		// UNACTIVATED -> active (if thread has been delegated a node to search); or UNACTIVATED -> DEACTIVATED (if all possible threads in the chain of delegators that could activate this thread have already become DEACTIVATED)
		if (search_code == StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::UNACTIVATED)
			StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::detNextIterState(search_ind,
																						search_inds_arr,
																						search_code,
																						search_codes_arr
																					);
	}
	// End processing loop
}
