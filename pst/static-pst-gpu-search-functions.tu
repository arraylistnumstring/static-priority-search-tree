#include "gpu-tree-node.h"
#include "warp-shuffles.h"

// Utilises dynamic parallelism
// Shared memory must be at least as large as (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Correctness only guaranteed for grids with one active block
// Non-member functions can only use at most one template clause
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void threeSidedSearchGlobal(T *const root_d, const size_t num_elem_slots,
										const size_t start_node_ind,
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

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(s);
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char *>(search_inds_arr + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;
	else
		search_inds_arr[threadIdx.x] = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
	// For threeSidedSearchGlobal(), each thread starts out with their code set to THREE_SEARCH
	search_codes_arr[threadIdx.x] = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::THREE_SEARCH;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];
	unsigned char search_code = search_codes_arr[threadIdx.x];

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;
	bool cell_active;

	while (cont_iter)
	{
		cell_active = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
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
				cell_active = min_dim1_val <= curr_node_dim1_val
								&& curr_node_dim1_val <= max_dim1_val;
			}
		}

		// Intrawarp shuffle; must be executed by all threads regardless of in/activity and search type
		unsigned long long res_ind_to_access = calcAllocReportIndOffset<false, unsigned long long>(cell_active);

		// Report step
		if (cell_active)
		{
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

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::THREE_SEARCH)	// Currently a three-sided query
			{
				// Do 3-sided search delegation
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::do3SidedSearchDelegation(curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d,
															min_dim1_val, max_dim1_val,
															curr_node_med_dim1_val, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr);
			}
			else if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH)
			{
				// Do left search delegation
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(curr_node_med_dim1_val <= max_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr);
			}
			else if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::RIGHT_SEARCH)
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doRightSearchDelegation(curr_node_med_dim1_val >= min_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr);
			}
			else	// search_code == REPORT_ALL
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAllNodesDelegation(curr_node_bitcode,
																root_d, num_elem_slots,
																res_arr_d, min_dim2_val,
																search_ind, search_inds_arr,
																search_codes_arr);
			}
		}


		__syncthreads();

		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detInactivity(search_ind, search_inds_arr, cont_iter, &search_code, search_codes_arr);
	}
	// End cont_iter loop
}

// Shared memory must be at least as large as (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Correctness only guaranteed for grids with one active block
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void twoSidedLeftSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const size_t start_node_ind,
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

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(s);
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char *>(search_inds_arr + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;
	else
		search_inds_arr[threadIdx.x] = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
	// For twoSidedLeftSearchGlobal(), all threads start with their search code set to LEFT_SEARCH
	search_codes_arr[threadIdx.x] = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];
	unsigned char search_code = search_codes_arr[threadIdx.x];

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;
	bool cell_active;

	while (cont_iter)
	{
		cell_active = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
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
				cell_active = curr_node_dim1_val <= max_dim1_val;
			}
		}

		// Intrawarp shuffle; must be executed by all threads regardless of in/activity and search type
		unsigned long long res_ind_to_access = calcAllocReportIndOffset<false, unsigned long long>(cell_active);
		
		// Report step
		if (cell_active)
		{
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

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH)	// Currently a search-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(curr_node_med_dim1_val <= max_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr);
			}
			else	// Already a report all-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAllNodesDelegation(curr_node_bitcode,
																root_d, num_elem_slots,
																res_arr_d, min_dim2_val,
																search_ind, search_inds_arr,
																search_codes_arr);
			}
		}

		__syncthreads();

		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detInactivity(search_ind, search_inds_arr, cont_iter, &search_code, search_codes_arr);
	}
	// End cont_iter loop
}

// Shared memory must be at least as large as (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Correctness only guaranteed for grids with one active block
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void twoSidedRightSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const size_t start_node_ind,
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

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(s);
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char *>(search_inds_arr + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;
	else
		search_inds_arr[threadIdx.x] = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;
	// For twoSidedRightSearchGlobal(), all threads start with their search code set to RIGHT_SEARCH
	search_codes_arr[threadIdx.x] = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::RIGHT_SEARCH;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];
	unsigned char search_code = search_codes_arr[threadIdx.x];

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;
	bool cell_active;

	while (cont_iter)
	{
		cell_active = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
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
				cell_active = curr_node_dim1_val >= min_dim1_val;
			}
		}

		// Intrawarp shuffle; must be executed by all threads regardless of in/activity and search type
		unsigned long long res_ind_to_access = calcAllocReportIndOffset<false, unsigned long long>(cell_active);

		// Report step
		if (cell_active)
		{
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

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (search_code == StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::RIGHT_SEARCH)	// Currently a search-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doRightSearchDelegation(curr_node_med_dim1_val >= min_dim1_val,
															curr_node_bitcode,
															root_d, num_elem_slots,
															res_arr_d, min_dim2_val,
															search_ind, search_inds_arr,
															search_code, search_codes_arr);
			}
			else	// Already a report all-type query
			{
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAllNodesDelegation(curr_node_bitcode,
																root_d, num_elem_slots,
																res_arr_d, min_dim2_val,
																search_ind, search_inds_arr,
																search_codes_arr);
			}
		}

		__syncthreads();


		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detInactivity(search_ind, search_inds_arr, cont_iter, &search_code, search_codes_arr);
	}
	// End cont_iter loop
}

// Shared memory must be at least as large as (number of threads) * sizeof(long long)
// Correctness only guaranteed for grids with one active block
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void reportAllNodesGlobal(T *const root_d, const size_t num_elem_slots,
										const size_t start_node_ind,
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

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(s);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = start_node_ind;
	else
		search_inds_arr[threadIdx.x] = StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set


	bool cont_iter = true;	// Loop-continuing flag

	long long search_ind = search_inds_arr[threadIdx.x];

	// curr_node_dim1_val will only be accessed once, so no need to create an automatic variable for it
	T curr_node_dim2_val;
	unsigned char curr_node_bitcode;
	bool cell_active;

	while (cont_iter)
	{
		cell_active = false;

		// active threads -> INACTIVE (if current node goes below the dim2_val threshold or has no children)
		// Before the next __syncthreads() call, which denotes the end of this section, active threads are the only threads who will modify their own search_inds_arr entry, so it is fine to do so non-atomically; also, this location is outside of the section where threads update each other's indices (which is blocked off by __syncthreads() calls), so it is extra safe
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
			{
				cell_active = true;
			}
		}

		// Intrawarp shuffle; must be executed by all threads regardless of in/activity and search type
		unsigned long long res_ind_to_access = calcAllocReportIndOffset<false, unsigned long long>(cell_active);

		if (cell_active)
		{
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

		// All threads who would become inactive in this iteration have finished; synchronisation is utilised because one must be certain that INACTIVE -> active writes (by other threads) are not inadvertently overwritten by active -> INACTIVE writes in lines of code above this one
		__syncthreads();


		// active threads -> active threads
		// INACTIVE threads -> active threads (external reactivation by active threads only)
		// If thread remains active, it must have at least one child
		if (search_ind != StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND)
		{
			if (GPUTreeNode::hasLeftChild(curr_node_bitcode)
					&& GPUTreeNode::hasRightChild(curr_node_bitcode))
			{
				// Delegate reporting of all nodes in right child to another thread and/or block
				StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::splitReportAllNodesWork(root_d, num_elem_slots,
															GPUTreeNode::getRightChild(search_ind),
															res_arr_d, min_dim2_val,
															search_inds_arr);

				// Prepare to report all nodes in the next iteration
				search_inds_arr[threadIdx.x] = search_ind
						= GPUTreeNode::getLeftChild(search_ind);
			}
			// Node only has a left child; report all on left child
			else if (GPUTreeNode::hasLeftChild(curr_node_bitcode))
			{
				search_inds_arr[threadIdx.x] = search_ind
					= GPUTreeNode::getLeftChild(search_ind);
			}
			// Node only has a right child; report all on right child
			else if (GPUTreeNode::hasRightChild(curr_node_bitcode))
			{
				search_inds_arr[threadIdx.x] = search_ind
					= GPUTreeNode::getRightChild(search_ind);
			}
		}

		__syncthreads();


		// No __syncthreads() call is necessary between detInactivity() and the end of the loop, as it can only potentially overlap with the section where active threads become inactive; this poses no issue for correctness, as if there is still work to be done, at least one thread will be guaranteed to remain active and therefore no inactive threads will exit the processing loop

		StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detInactivity(search_ind, search_inds_arr, cont_iter);
	}
	// End cont_iter loop
}
