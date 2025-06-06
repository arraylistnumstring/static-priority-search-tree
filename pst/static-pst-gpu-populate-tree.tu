// For use of std::move(); CUDA automatically gives it __host__ __device__ qualifiers, unless explicitly specified against during compilation (CUDA Programming Guide 14.5.22.3)
#include <utility>

#include "gpu-tree-node.h"


// Utilises dynamic parallelism
// Shared memory must be at least as large as (total number of threads) * sizeof(size_t) * num_constr_working_arrs (currently 3)
// Correctness only guaranteed for grids with one active block
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__global__ void populateTree(T *const root_d, const size_t num_elem_slots,
								PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
								size_t *const dim1_val_ind_arr_d, size_t *dim2_val_ind_arr_d,
								size_t *dim2_val_ind_arr_secondary_d,
								const size_t val_ind_arr_start_ind, const size_t num_elems,
								const size_t target_node_start_ind)
{
	// For correctness, only 1 block can ever be active, as synchronisation across blocks (i.e. global synchronisation) is not possible without exiting the kernel entirely
	if (blockIdx.x != 0)
		return;

	cooperative_groups::thread_block curr_block = cooperative_groups::this_thread_block();

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	size_t *subelems_start_inds_arr = reinterpret_cast<size_t *>(s);
	size_t *num_subelems_arr = subelems_start_inds_arr + blockDim.x;
	size_t *target_node_inds_arr = num_subelems_arr + blockDim.x;
	// Initialise shared memory
	subelems_start_inds_arr[threadIdx.x] = val_ind_arr_start_ind;
	// All threads except for thread 0 start by being inactive
	num_subelems_arr[threadIdx.x] = threadIdx.x == 0 ? num_elems : 0;
	target_node_inds_arr[threadIdx.x] = target_node_start_ind;

	curr_block.sync();

	// To minimise the total number of dynamic parallelism kernel launches, after the threads fully branch out into a subtree (with each thread constructing one node), for the next iteration, have the threads recongregate at thread ID 0's new node in the tree and repeat the process
	while (*num_subelems_arr > 0)
	{
		// At any given level of the tree, each thread creates one node; as depth h of a tree has 2^h nodes, the number of active threads at level h is 2^h
		// num_subelems_arr[threadIdx.x] > 0 condition not written here, as almost all threads except for thread 0 start out with the value num_subelems_arr[threadIdx.x] == 0
#pragma unroll
		for (size_t nodes_per_level = 1; nodes_per_level <= minPowerOf2AtLeast(blockDim.x);
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
				long long array_search_res_ind = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::binarySearch(pt_arr_d, dim1_val_ind_arr_d,
													pt_arr_d[dim2_val_ind_arr_d[subelems_start_inds_arr[threadIdx.x]]],
													subelems_start_inds_arr[threadIdx.x],
													num_subelems_arr[threadIdx.x]);

				// Check to make sure array_search_res_ind is a valid index (which it should be, since it's an element of pt_arr to begin with)
				assert(array_search_res_ind != -1);

				// Note: potential sign conversion issue when computer memory becomes of size 2^64
				const size_t max_dim2_val_dim1_array_ind = array_search_res_ind;

				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::constructNode(root_d, num_elem_slots,
											pt_arr_d, target_node_inds_arr[threadIdx.x],
											dim1_val_ind_arr_d, dim2_val_ind_arr_d,
												dim2_val_ind_arr_secondary_d,
												max_dim2_val_dim1_array_ind,
											subelems_start_inds_arr[threadIdx.x],
											num_subelems_arr[threadIdx.x],
											left_subarr_num_elems, right_subarr_start_ind,
											right_subarr_num_elems);

				// Update information for next iteration; as memory accesses are coalesced no matter the relative order as long as they are from the same source location, (and nodes are consecutive except possibly at the leaf levels), pick an inactive thread to instantiate the right child
				// If there exist inactive threads in the block, assign the right child to an inactive thread and the left child to oneself
				if (threadIdx.x + nodes_per_level < blockDim.x)
				{
					subelems_start_inds_arr[threadIdx.x + nodes_per_level] = right_subarr_start_ind;
					num_subelems_arr[threadIdx.x + nodes_per_level] = right_subarr_num_elems;
					target_node_inds_arr[threadIdx.x + nodes_per_level] =
						GPUTreeNode::getRightChild(target_node_inds_arr[threadIdx.x]);

					num_subelems_arr[threadIdx.x] = left_subarr_num_elems;
					target_node_inds_arr[threadIdx.x] =
						GPUTreeNode::getLeftChild(target_node_inds_arr[threadIdx.x]);
				}
				// Dynamic parallelism on the children of the level of the tree where all threads are active, or the one above, if blockDim.x is not a power of 2, and some threads have no existing threads to which to assign the children of its current node
				else
				{
					// Primary and secondary arrays must be swapped at the next level
					/*
						Note on dynamic parallelism:
							Original goal: use cudaStreamFireAndForget for maximal kernel concurrency
							However, with sufficiently large data (e.g. n = 128^3 = (2^7)^3 = 2^21 elements), cudaStreamFireAndForget causes incorrect initialisations of tree nodes (to (0, 0; 0; 0)), even when device-code optimisations are turned off and an attempt at keeping track of the number of active grids is applied (when device-code optimisations are turned on, further initialisation errors occur).
							Given the default kernel queueing limit of 2^11 = 2048 kernels, this is clearly an overshoot of the hardware's capabilities by a factor of about 2^10 = 1024 (also of note: hardware only supports at most 2^7 = 128 concurrent kernels, so the hardware is not designed to rely on dynamic parallelism for the bulk of its concurrency); hence, for code correctness at large problem sizes, the default stream (value NULL; is also achieved when unspecified) must be used for construction-side dynamic parallelism, though this causes a massive hit to the performance of tree construction.

							For record-keeping purposes:
							First iteration of dynamic parallelism sent all dynamic parallelism kernel calls to cudaStreamFireAndForget, which caused launch failures and left tree branches uninitialised at lower levels, with only 0 or otherwise garbage values
							Second iteration of dynamic parallelism was an attempt to keep track of the total number of active grids and dynamically send to the default stream and/or cudaStreamFireAndForget depending on whether the number of active grids was below the hardware-set threshold; this issue still causes incorrect initialisations of tree nodes, though compute-sanitizer did not complain in this case with certain problem sizes that 
								Switch in question: 
									(num_active_grids_d < MAX_NUM_ACTIVE_GRIDS) ?
										cudaStreamFireAndForget : NULL
									Symbols used above were defined in helpers/dev-symbols.h

								In both the first and second cases, errors in compute-sanitizer arising from cudaStreamFireAndForget present either as memory writes beyond allocated space (presumably when trying to write to the kernel queue) or as "unspecified launch failure"s; increasing the size of the kernel queue only postpones such problems

							Third (current) iteration: default stream: handles large problem sizes, whether code is compiled with or without optimisations
					*/
					populateTree<<<1, blockDim.x, blockDim.x * sizeof(size_t)
										* StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::num_constr_working_arrs>>>
						(root_d, num_elem_slots, pt_arr_d, dim1_val_ind_arr_d,
							dim2_val_ind_arr_secondary_d, dim2_val_ind_arr_d,
							right_subarr_start_ind, right_subarr_num_elems,
							GPUTreeNode::getRightChild(target_node_inds_arr[threadIdx.x]));

					// If all threads were active, have all threads with ID other than 0 delegate their left child to a new grid
					if (threadIdx.x != 0 && nodes_per_level >= blockDim.x)
					{
						populateTree<<<1, blockDim.x, blockDim.x * sizeof(size_t)
											* StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::num_constr_working_arrs>>>
							(root_d, num_elem_slots, pt_arr_d, dim1_val_ind_arr_d,
								dim2_val_ind_arr_secondary_d, dim2_val_ind_arr_d,
								subelems_start_inds_arr[threadIdx.x],
								left_subarr_num_elems,
								GPUTreeNode::getLeftChild(target_node_inds_arr[threadIdx.x]));

						// Reset to 0 in preparation for next iteration of branching construction
						num_subelems_arr[threadIdx.x] = 0;
					}
					// If not all threads were active (because blockDim.x is not a power of 2, but there were no more threads for this thread to delegate to), or if thread has ID 0, move to left child of current node
					else
					{
						num_subelems_arr[threadIdx.x] = left_subarr_num_elems;
						target_node_inds_arr[threadIdx.x] =
							GPUTreeNode::getLeftChild(target_node_inds_arr[threadIdx.x]);
					}
				}
			}

			/*
				Only one curr_block.sync() call is needed in the loop because inactive threads do nothing to modify their own state-control variables (specifically, affecting neither their in/activity nor their continued participation in the loop); they only swap their local dimension-2 index array pointers
				Additionally, a curr_block.sync() call at the end of an iteration effectively serves as a curr_block.sync() call at the beginning of the loop as well (as does the initial curr_block.sync() call before the loop), preventing any given iteration (and its associated updates to shared memory) from impacting any other iteration (where said info would be read) incorrectly
			*/
			cooperative_groups::thread_block::arrival_token process_curr_node_arrival_token = curr_block.barrier_arrive();

			// Every thread must swap its primary and secondary dim2_val_ind_arr pointers in order to have the correct subordering of indices at a given node
			size_t *temp = dim2_val_ind_arr_d;
			dim2_val_ind_arr_d = dim2_val_ind_arr_secondary_d;
			dim2_val_ind_arr_secondary_d = temp;

			// Synchronise before starting the next iteration so that inactive threads that should be activated will know so
			curr_block.barrier_wait(std::move(process_curr_node_arrival_token));
		}
	}
}
