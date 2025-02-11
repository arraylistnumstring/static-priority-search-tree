#include <algorithm>	// To use sort()
#include <cstring>		// To use memcpy()
#include <string>		// To use string-building functions

#include "err-chk.h"
#include "resize-array.h"

#ifdef CONSTR_TIMED
#include <chrono>		// To use std::chrono::steady_clock (a monotonic clock suitable for interval measurements) and related functions
#include <ctime>		// To use std::clock_t (CPU timer; pauses when CPU pauses, etc.)
#endif

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::StaticPSTCPUIter(PointStructTemplate<T, IDType, num_IDs> *const pt_arr, size_t num_elems)
	// Member initialiser list must be followed by definition
	// Number of element slots in each container subarray is nextGreaterPowerOf2(num_elems) - 1
	: num_elem_slots(num_elems == 0 ? 0 : nextGreaterPowerOf2(num_elems) - 1),
	num_elems(num_elems)
{
	if (num_elems == 0)
	{
		root = nullptr;
		return;
	}

	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	// constexpr if is a C++17 feature that only compiles the branch of code that evaluates to true at compile-time, saving executable space and execution runtime
	// Use of !HasID<> serves as an effective guard against IDType=void instantiations
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
	{
		// No IDs present or sizeof(T) >= sizeof(IDType)
		size_t tot_arr_size_num_Ts = calcTotArrSizeNumTs<num_val_subarrs>(num_elem_slots);
		root = new T[tot_arr_size_num_Ts]();
	}
	else
	{
		// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
		size_t tot_arr_size_num_IDTypes = calcTotArrSizeNumIDTypes<num_val_subarrs>(num_elem_slots);
		root = reinterpret_cast<T *>(new IDType[tot_arr_size_num_IDTypes]());
	}

	if (root == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type " + typeid(T).name() + " to root");

	// Create two arrays of PointStruct indices for processing PointStruct objects
	size_t *dim1_val_ind_arr = new size_t[num_elems];
	if (dim1_val_ind_arr == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type size_t to dim1_val_ind_arr");
	size_t *dim2_val_ind_arr = new size_t[num_elems];
	if (dim2_val_ind_arr == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type size_t to dim2_val_ind_arr");
	size_t *dim2_val_ind_arr_secondary = new size_t[num_elems];
	if (dim2_val_ind_arr_secondary == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type size_t to dim2_val_ind_arr_secondary");

#ifdef CONSTR_TIMED
	std::clock_t ind_assign_start_CPU, ind_assign_stop_CPU;
	std::chrono::time_point<std::chrono::steady_clock> ind_assign_start_wall, ind_assign_stop_wall;

	std::clock_t ind1_sort_start_CPU, ind1_sort_stop_CPU;
	std::chrono::time_point<std::chrono::steady_clock> ind1_sort_start_wall, ind1_sort_stop_wall;

	std::clock_t ind2_sort_start_CPU, ind2_sort_stop_CPU;
	std::chrono::time_point<std::chrono::steady_clock> ind2_sort_start_wall, ind2_sort_stop_wall;

	std::clock_t populate_tree_start_CPU, populate_tree_stop_CPU;
	std::chrono::time_point<std::chrono::steady_clock> populate_tree_start_wall, populate_tree_stop_wall;

	ind_assign_start_CPU = std::clock();
	ind_assign_start_wall = std::chrono::steady_clock::now();
#endif

	for (size_t i = 0; i < num_elems; i++)
		dim1_val_ind_arr[i] = dim2_val_ind_arr[i] = i;

#ifdef CONSTR_TIMED
	ind_assign_stop_CPU = std::clock();
	ind_assign_stop_wall = std::chrono::steady_clock::now();

	ind1_sort_start_CPU = std::clock();
	ind1_sort_start_wall = std::chrono::steady_clock::now();
#endif

	// Sort dimension-1 values index array in ascending order; in-place sort using a curried comparison function
	std::sort(dim1_val_ind_arr, dim1_val_ind_arr + num_elems,
				[](PointStructTemplate<T, IDType, num_IDs> *const &pt_arr)
					{
						// [&] captures all variables in enclosing scope by reference so that they can be used within the body of the lambda function
						return [&](const size_t &i, const size_t &j)
							{
								return pt_arr[i].compareDim1(pt_arr[j]) < 0;
							};
					}(pt_arr));	// Parentheses immediately after a lambda definition serves to call it with the given parameter

#ifdef CONSTR_TIMED
	ind1_sort_stop_CPU = std::clock();
	ind1_sort_stop_wall = std::chrono::steady_clock::now();

	ind2_sort_start_CPU = std::clock();
	ind2_sort_start_wall = std::chrono::steady_clock::now();
#endif

	// Sort dimension-2 values index array in descending order; in-place sort using a curried comparison function
	std::sort(dim2_val_ind_arr, dim2_val_ind_arr + num_elems,
				[](PointStructTemplate<T, IDType, num_IDs> *const &pt_arr)
					{
						return [&](const size_t &i, const size_t &j)
							{
								return pt_arr[i].compareDim2(pt_arr[j]) > 0;
							};
					}(pt_arr));

#ifdef CONSTR_TIMED
	ind2_sort_stop_CPU = std::clock();
	ind2_sort_stop_wall = std::chrono::steady_clock::now();
#endif

#ifdef DEBUG
	/*
		Code that is only compiled when debugging; to define this preprocessor variable, compile with the option -DDEBUG, as in
			nvcc -DDEBUG <source-code-file>

		Explicitly printed to sanity-check the corresponding code in StaticPSTGPU
	*/
	const size_t cudaWarpSize = 32;
	std::cout << "Would call populateTree() with " << 1 << " block, " << nextGreaterPowerOf2(cudaWarpSize - 1) << " threads, " << (nextGreaterPowerOf2(cudaWarpSize - 1) * sizeof(size_t) << 1) << " B of shared memory\n";
#endif

#ifdef CONSTR_TIMED
	populate_tree_start_CPU = std::clock();
	populate_tree_start_wall = std::chrono::steady_clock::now();
#endif

	populateTree(root, num_elem_slots, pt_arr, dim1_val_ind_arr, dim2_val_ind_arr, dim2_val_ind_arr_secondary, 0, num_elems);

#ifdef CONSTR_TIMED
	populate_tree_stop_CPU = std::clock();
	populate_tree_stop_wall = std::chrono::steady_clock::now();

	std::cout << "CPU PST index assignment time:\n"
			  << "\tCPU clock time used:\t"
			  << 1000.0 * (ind_assign_stop_CPU - ind_assign_start_CPU) / CLOCKS_PER_SEC << " ms\n"
			  << "\tWall clock time passed:\t"
			  << std::chrono::duration<double, std::milli>(ind_assign_stop_wall - ind_assign_start_wall).count()
			  << " ms\n";

	std::cout << "CPU PST index dimension-1-based sorting time:\n"
			  << "\tCPU clock time used:\t"
			  << 1000.0 * (ind1_sort_stop_CPU - ind1_sort_start_CPU) / CLOCKS_PER_SEC << " ms\n"
			  << "\tWall clock time passed:\t"
			  << std::chrono::duration<double, std::milli>(ind1_sort_stop_wall - ind1_sort_start_wall).count()
			  << " ms\n";

	std::cout << "CPU PST index dimension-2-based sorting time:\n"
			  << "\tCPU clock time used:\t"
			  << 1000.0 * (ind2_sort_stop_CPU - ind2_sort_start_CPU) / CLOCKS_PER_SEC << " ms\n"
			  << "\tWall clock time passed:\t"
			  << std::chrono::duration<double, std::milli>(ind2_sort_stop_wall - ind2_sort_start_wall).count()
			  << " ms\n";

	std::cout << "CPU PST tree-population code time:\n"
			  << "\tCPU clock time used:\t"
			  << 1000.0 * (populate_tree_stop_CPU - populate_tree_start_CPU) / CLOCKS_PER_SEC << " ms\n"
			  << "\tWall clock time passed:\t"
			  << std::chrono::duration<double, std::milli>(populate_tree_stop_wall - populate_tree_start_wall).count()
			  << " ms\n";
#endif

	delete[] dim1_val_ind_arr;
	delete[] dim2_val_ind_arr;
	delete[] dim2_val_ind_arr_secondary;
}

// const keyword after method name indicates that the method does not modify any data members of the associated class
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
	if (num_elems == 0)
	{
		os << "Tree is empty\n";
		return;
	}

	std::string prefix = "";
	std::string child_prefix = "";
	printRecur(os, root, 0, num_elem_slots, prefix, child_prefix);
}

// Separate template clauses are necessary when the enclosing template class has different template types from the member function
// Default template argument for a class template's member function can only be specified within the class template
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::threeSidedSearch(size_t &num_res_elems, RetType *&res_arr, T min_dim1_val, T max_dim1_val, T min_dim2_val)
{
	if (num_elems == 0)
	{
		std::cout << "Tree is empty; nothing to search\n";
		num_res_elems = 0;
		res_arr = nullptr;
		return;
	}

	size_t res_arr_size = num_elems;
	res_arr = new RetType[res_arr_size];
	num_res_elems = 0;

	std::stack<long long> search_inds_stack;
	std::stack<unsigned char> search_codes_stack;

	search_inds_stack.push(0);
	search_codes_stack.push(THREE_SEARCH);

	long long search_ind;
	unsigned char search_code;

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;

	// Stacks are synchronised, so 1 stack is empty exactly when both are
	while (!search_inds_stack.empty())
	{
		search_ind = search_inds_stack.top();
		search_inds_stack.pop();
		search_code = search_codes_stack.top();
		search_codes_stack.pop();

		// Note that because this program is single-threaded, there is never a chance of inactivity, as the lone thread only terminates when there is no more work to be done; hence, there is no INACTIVE_IND check (when comparing to the analogous GPU version)
		curr_node_dim1_val = getDim1ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_dim2_val = getDim2ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_med_dim1_val = getMedDim1ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_bitcode = getBitcodesRoot(root, num_elem_slots)[search_ind];

		// Only process node if its dimension-2 value satisfies the dimension-2 search bound
		if (min_dim2_val <= curr_node_dim2_val)
		{
			// Check if current node satisfies query and should be reported
			if (min_dim1_val <= curr_node_dim1_val
					&& curr_node_dim1_val <= max_dim1_val)
			{
				if constexpr (std::is_same<RetType, IDType>::value)
					res_arr[num_res_elems] = getIDsRoot(root, num_elem_slots)[search_ind];
				else
				{
					res_arr[num_res_elems].dim1_val = curr_node_dim1_val;
					res_arr[num_res_elems].dim2_val = curr_node_dim2_val;
					// As IDs are only accessed if the node is to be reported and if IDs exist, don't waste a register on it (and avoid compilation failures from attempting to instantiate a potential void variable)
					if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
						res_arr[num_res_elems].id = getIDsRoot(root, num_elem_slots)[search_ind];
				}
				num_res_elems++;
			}

			// Delegation/further activity down this branch necessary only if this node has children and can therefore be searched
			if (CPUIterTreeNode::hasChildren(curr_node_bitcode))
			{
				if (search_code == THREE_SEARCH)	// Currently a three-sided query
				{
					do3SidedSearchDelegation(curr_node_bitcode,
												min_dim1_val, max_dim1_val,
												curr_node_med_dim1_val,
												search_ind, search_inds_stack,
												search_codes_stack);
				}
				else if (search_code == LEFT_SEARCH)
				{
					doLeftSearchDelegation(curr_node_med_dim1_val <= max_dim1_val,
											curr_node_bitcode,
											search_ind, search_inds_stack,
											search_codes_stack);
				}
				else if (search_code == RIGHT_SEARCH)
				{
					doRightSearchDelegation(curr_node_med_dim1_val >= min_dim1_val,
											curr_node_bitcode,
											search_ind, search_inds_stack,
											search_codes_stack);
				}
				else	// Already a report all-type query
				{
					doReportAllNodesDelegation(curr_node_bitcode,
												search_ind, search_inds_stack,
												search_codes_stack);
				}
			}
		}
	}

	// Ensure that no more memory is taken up than needed
	if (res_arr_size > num_res_elems)
		resizeArray(res_arr, res_arr_size, num_res_elems);
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::twoSidedLeftSearch(size_t &num_res_elems, RetType *&res_arr, T max_dim1_val, T min_dim2_val)
{
	if (num_elems == 0)
	{
		std::cout << "Tree is empty; nothing to search\n";
		num_res_elems = 0;
		res_arr = nullptr;
		return;
	}

	size_t res_arr_size = num_elems;
	res_arr = new RetType[res_arr_size];
	num_res_elems = 0;

	std::stack<long long> search_inds_stack;
	std::stack<unsigned char> search_codes_stack;

	search_inds_stack.push(0);
	search_codes_stack.push(LEFT_SEARCH);

	long long search_ind;
	unsigned char search_code;

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;

	// Stacks are synchronised, so 1 stack is empty exactly when both are
	while (!search_inds_stack.empty())
	{
		search_ind = search_inds_stack.top();
		search_inds_stack.pop();
		search_code = search_codes_stack.top();
		search_codes_stack.pop();

		// Note that because this program is single-threaded, there is never a chance of inactivity, as the lone thread only terminates when there is no more work to be done; hence, there is no INACTIVE_IND check (when comparing to the analogous GPU version)
		curr_node_dim1_val = getDim1ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_dim2_val = getDim2ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_med_dim1_val = getMedDim1ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_bitcode = getBitcodesRoot(root, num_elem_slots)[search_ind];

		// Only process node if its dimension-2 value satisfies the dimension-2 search bound
		if (min_dim2_val <= curr_node_dim2_val)
		{
			// Check if current node staisfies query and should be reported
			if (curr_node_dim1_val <= max_dim1_val)
			{
				if constexpr (std::is_same<RetType, IDType>::value)
					res_arr[num_res_elems] = getIDsRoot(root, num_elem_slots)[search_ind];
				else
				{
					res_arr[num_res_elems].dim1_val = curr_node_dim1_val;
					res_arr[num_res_elems].dim2_val = curr_node_dim2_val;
					if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
						res_arr[num_res_elems].id = getIDsRoot(root, num_elem_slots)[search_ind];
				}
				num_res_elems++;
			}

			// Delegation/further activity down this branch necessary only if this node has children and can therefore be searched
			if (CPUIterTreeNode::hasChildren(curr_node_bitcode))
			{
				if (search_code == LEFT_SEARCH)
				{
					doLeftSearchDelegation(curr_node_med_dim1_val <= max_dim1_val,
											curr_node_bitcode,
											search_ind, search_inds_stack,
											search_codes_stack);
				}
				else	// Already a report all-type query
				{
					doReportAllNodesDelegation(curr_node_bitcode,
												search_ind, search_inds_stack,
												search_codes_stack);
				}
			}
		}
	}

	// Ensure that no more memory is taken up than needed
	if (res_arr_size > num_res_elems)
		resizeArray(res_arr, res_arr_size, num_res_elems);
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::twoSidedRightSearch(size_t &num_res_elems, RetType *&res_arr, T min_dim1_val, T min_dim2_val)
{
	if (num_elems == 0)
	{
		std::cout << "Tree is empty; nothing to search\n";
		num_res_elems = 0;
		res_arr = nullptr;
		return;
	}

	size_t res_arr_size = num_elems;
	res_arr = new RetType[res_arr_size];
	num_res_elems = 0;

	std::stack<long long> search_inds_stack;
	std::stack<unsigned char> search_codes_stack;

	search_inds_stack.push(0);
	search_codes_stack.push(RIGHT_SEARCH);

	long long search_ind;
	unsigned char search_code;

	T curr_node_dim1_val;
	T curr_node_dim2_val;
	T curr_node_med_dim1_val;
	unsigned char curr_node_bitcode;

	// Stacks are synchronised, so 1 stack is empty exactly when both are
	while (!search_inds_stack.empty())
	{
		search_ind = search_inds_stack.top();
		search_inds_stack.pop();
		search_code = search_codes_stack.top();
		search_codes_stack.pop();

		// Note that because this program is single-threaded, there is never a chance of inactivity, as the lone thread only terminates when there is no more work to be done; hence, there is no INACTIVE_IND check (when comparing to the analogous GPU version)
		curr_node_dim1_val = getDim1ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_dim2_val = getDim2ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_med_dim1_val = getMedDim1ValsRoot(root, num_elem_slots)[search_ind];
		curr_node_bitcode = getBitcodesRoot(root, num_elem_slots)[search_ind];

		// Only process node if its dimension-2 value satisfies the dimension-2 search bound
		if (min_dim2_val <= curr_node_dim2_val)
		{
			// Check if current node satisfies query and should be reported
			if (curr_node_dim1_val >= min_dim1_val)
			{
				if constexpr (std::is_same<RetType, IDType>::value)
					res_arr[num_res_elems] = getIDsRoot(root, num_elem_slots)[search_ind];
				else
				{
					res_arr[num_res_elems].dim1_val = curr_node_dim1_val;
					res_arr[num_res_elems].dim2_val = curr_node_dim2_val;
					if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
						res_arr[num_res_elems].id = getIDsRoot(root, num_elem_slots)[search_ind];
				}
				num_res_elems++;
			}

			// Delegation/further activity down this branch necessary only if this node has children and can therefore be searched
			if (CPUIterTreeNode::hasChildren(curr_node_bitcode))
			{
				if (search_code == RIGHT_SEARCH)
				{
					doRightSearchDelegation(curr_node_med_dim1_val >= min_dim1_val,
											curr_node_bitcode,
											search_ind, search_inds_stack,
											search_codes_stack);
				}
				else	// Already a report all-type query
				{
					doReportAllNodesDelegation(curr_node_bitcode,
												search_ind, search_inds_stack,
												search_codes_stack);
				}
			}
		}
	}

	// Ensure that no more memory is taken up than needed
	if (res_arr_size > num_res_elems)
		resizeArray(res_arr, res_arr_size, num_res_elems);
}

// static keyword should only be used when declaring a function in the header file
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::constructNode(T *const root,
										const size_t num_elem_slots,
										PointStructTemplate<T, IDType, num_IDs> *const pt_arr,
										const size_t target_node_ind,
										const size_t num_elems,
										size_t *const dim1_val_ind_arr,
										size_t *const dim2_val_ind_arr,
										size_t *const dim2_val_ind_arr_secondary,
										const size_t max_dim2_val_dim1_array_ind,
										std::stack<size_t> &subelems_start_inds,
										std::stack<size_t> &num_subelems_stack,
										size_t &left_subarr_num_elems,
										size_t &right_subarr_start_ind,
										size_t &right_subarr_num_elems)
{
	size_t median_dim1_val_ind;

	// Treat dim1_val_ind_arr_ind[max_dim2_val_dim1_array_ind] as a removed element (but don't actually remove the element for performance reasons

	if (num_subelems_stack.top() == 1)		// Base case
	{
		median_dim1_val_ind = subelems_start_inds.top();
		left_subarr_num_elems = 0;
		right_subarr_num_elems = 0;
	}
	// max_dim2_val originally comes from the part of the array to the left of median_dim1_val
	else if (max_dim2_val_dim1_array_ind < subelems_start_inds.top() + num_subelems_stack.top()/2)
	{
		// As median values are always put in the left subtree, when the subroot value comes from the left subarray, the median index is given by num_elems/2, which evenly splits the array if there are an even number of elements remaining or makes the left subtree larger by one element if there are an odd number of elements remaining
		median_dim1_val_ind = subelems_start_inds.top()
								+ num_subelems_stack.top()/2;
		// max_dim2_val has been removed from the left subarray, so there are median_dim1_val_ind elements remaining on the left side
		left_subarr_num_elems = median_dim1_val_ind - subelems_start_inds.top();
		right_subarr_num_elems = subelems_start_inds.top() + num_subelems_stack.top()
									- median_dim1_val_ind - 1;
	}
	/*
		max_dim2_val originally comes from the part of the array to the right of median_dim1_val, i.e.
			max_dim2_val_dim1_array_ind >= subelems_start_inds.top() + num_subelems_stack.top()/2
	*/
	else
	{
		median_dim1_val_ind = subelems_start_inds.top()
								+ num_subelems_stack.top()/2 - 1;

		left_subarr_num_elems = median_dim1_val_ind - subelems_start_inds.top() + 1;
		right_subarr_num_elems = subelems_start_inds.top() + num_subelems_stack.top()
									- median_dim1_val_ind - 2;
	}

	setNode(root, target_node_ind, num_elem_slots,
			pt_arr[dim2_val_ind_arr[subelems_start_inds.top()]],
			pt_arr[dim1_val_ind_arr[median_dim1_val_ind]].dim1_val);

	if (left_subarr_num_elems > 0)
	{
		CPUIterTreeNode::setLeftChild(getBitcodesRoot(root, num_elem_slots), target_node_ind);

		// Always place median value in left subtree
		// max_dim2_val is to the left of median_dim1_val in dim1_val_ind_arr; shift all entries up to median_dim1_val_ind leftward, overwriting max_dim2_val_dim1_array_ind
		if (max_dim2_val_dim1_array_ind < median_dim1_val_ind)
			// memcpy() is undefined if the source and destination regions overlap, and the safe memmove() (that behaves as if the source values were first copied to an intermediate array) does not exist on CUDA within kernel code
			for (size_t i = max_dim2_val_dim1_array_ind; i < median_dim1_val_ind; i++)
				dim1_val_ind_arr[i] = dim1_val_ind_arr[i+1];
		// Otherwise, max_dim2_val is to the right of median_dim1_val_ind in dim1_val_ind_arr; leave left subarray as is
	}
	if (right_subarr_num_elems > 0)
	{
		CPUIterTreeNode::setRightChild(getBitcodesRoot(root, num_elem_slots), target_node_ind);

		// max_dim2_val is to the right of median_dim1_val_ind in dim1_val_ind_arr; shift all entries after max_dim2_val_dim1_array_ind leftward, overwriting max_dim2_val_array_ind
		if (max_dim2_val_dim1_array_ind > median_dim1_val_ind)
			for (size_t i = max_dim2_val_dim1_array_ind;
					i < subelems_start_inds.top() + num_subelems_stack.top() - 1;
					i++)
				dim1_val_ind_arr[i] = dim1_val_ind_arr[i+1];
		// Otherwise, max_dim2_val is to the left of median_dim1_val in dim1_val_ind_arr; leave right subarray as is
	}


	// Choose median_dim1_val_ind + 1 as the starting index for all right subarrays, as this is the only index that is valid no matter whether max_dim2_val is to the left or right of med_dim1_val
	right_subarr_start_ind = median_dim1_val_ind + 1;


	// Iterate through dim2_val_ind_arr, placing data with lower dimension-1 value in the subarray for the left subtree and data with higher dimension-1 value in the subarray for the right subtree
	// Note that because subarrays' sizes were allocated based on this same data, there should not be a need to check that left_dim2_subarr_iter_ind < left_subarr_num_elems (and similarly for the right side)
	if (CPUIterTreeNode::hasChildren(getBitcodesRoot(root, num_elem_slots)[target_node_ind]))
	{
		size_t left_dim2_subarr_iter_ind = subelems_start_inds.top();
		size_t right_dim2_subarr_iter_ind = right_subarr_start_ind;

		// Skip over first (largest) element in dim2_val_ind_arr, as it has already been placed in the current node
		for (size_t i = subelems_start_inds.top() + 1;
				i < subelems_start_inds.top() + num_subelems_stack.top();
				i++)
		{
			// dim2_val_ind_arr[i] is the index of a PointStruct that comes before or is the PointStruct of median dim1 value in dim1_val_ind_arr
			if (pt_arr[dim2_val_ind_arr[i]].compareDim1(pt_arr[dim1_val_ind_arr[median_dim1_val_ind]]) <= 0)
				// Postfix ++ returns the current value before incrementing
				dim2_val_ind_arr_secondary[left_dim2_subarr_iter_ind++] = dim2_val_ind_arr[i];
			// dim2_val_ind_arr[i] is the index of a PointStruct that comes after the PointStruct of median dim1 value in dim1_val_ind_arr
			else
				dim2_val_ind_arr_secondary[right_dim2_subarr_iter_ind++] = dim2_val_ind_arr[i];
		}
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::do3SidedSearchDelegation(
											const unsigned char curr_node_bitcode,
											T min_dim1_val, T max_dim1_val,
											T curr_node_med_dim1_val,
											const long long search_ind,
											std::stack<long long> &search_inds_stack,
											std::stack<unsigned char> &search_codes_stack
										)
{
	// Splitting of query is only possible if the current node has two children and min_dim1_val <= curr_node_med_dim1_val <= max_dim1_val; the equality on max_dim1_val is for the edge case where a median point may be duplicated, with one copy going to the left subtree and the other to the right subtree
	if (min_dim1_val <= curr_node_med_dim1_val
			&& curr_node_med_dim1_val <= max_dim1_val)
	{
		// Query splits over median; split into 2 two-sided queries
		// Search left subtree with a two-sided right search
		if (CPUIterTreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_stack.push(CPUIterTreeNode::getLeftChild(search_ind));
			search_codes_stack.push(RIGHT_SEARCH);
		}
		// Search right subtree with a two-sided left search
		if (CPUIterTreeNode::hasRightChild(curr_node_bitcode))
		{
			search_inds_stack.push(CPUIterTreeNode::getRightChild(search_ind));
			search_codes_stack.push(LEFT_SEARCH);
		}
	}
	// Perform three-sided search on left child
	else if (max_dim1_val < curr_node_med_dim1_val
				&& CPUIterTreeNode::hasLeftChild(curr_node_bitcode))
	{
		search_inds_stack.push(CPUIterTreeNode::getLeftChild(search_ind));
		search_codes_stack.push(THREE_SEARCH);
	}
	// Perform three-sided search on right child
	else if (curr_node_med_dim1_val < min_dim1_val
				&& CPUIterTreeNode::hasRightChild(curr_node_bitcode))
	{
		search_inds_stack.push(CPUIterTreeNode::getRightChild(search_ind));
		search_codes_stack.push(THREE_SEARCH);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(
											const bool range_split_poss,
											const unsigned char curr_node_bitcode,
											const long long search_ind,
											std::stack<long long> &search_inds_stack,
											std::stack<unsigned char> &search_codes_stack
										)
{
	// Report all nodes in left subtree, "recurse" search on right
	// Though the upper bound of the dimension-1 search range is typically open, if there are duplicates of the median point and one happens to be allocated to each subtree, both trees must be traversed for correctness
	if (range_split_poss)
	{
		// If current node has left child, report all on left child
		if (CPUIterTreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_stack.push(CPUIterTreeNode::getLeftChild(search_ind));
			search_codes_stack.push(REPORT_ALL);
		}
		// If current node has right child, search right child
		if (CPUIterTreeNode::hasRightChild(curr_node_bitcode))
		{
			search_inds_stack.push(CPUIterTreeNode::getRightChild(search_ind));
			search_codes_stack.push(LEFT_SEARCH);
		}
	}
	// !range_split_poss
	// Only left subtree can possibly contain valid entries; search left subtree
	else if (CPUIterTreeNode::hasLeftChild(curr_node_bitcode))
	{
		search_inds_stack.push(CPUIterTreeNode::getLeftChild(search_ind));
		search_codes_stack.push(LEFT_SEARCH);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::doRightSearchDelegation(
											const bool range_split_poss,
											const unsigned char curr_node_bitcode,
											const long long search_ind,
											std::stack<long long> &search_inds_stack,
											std::stack<unsigned char> &search_codes_stack
										)
{
	// Report all nodes in right subtree, "recurse" search on left
	if (range_split_poss)
	{
		// If current node has left child, search left child
		if (CPUIterTreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_stack.push(CPUIterTreeNode::getLeftChild(search_ind));
			search_codes_stack.push(RIGHT_SEARCH);
		}
		// If current node has right child, report all on right child
		if (CPUIterTreeNode::hasRightChild(curr_node_bitcode))
		{
			search_inds_stack.push(CPUIterTreeNode::getRightChild(search_ind));
			search_codes_stack.push(REPORT_ALL);
		}
	}
	// !range_split_poss
	// Only right subtree can possibly contain valid entries; search right subtree
	else if (CPUIterTreeNode::hasRightChild(curr_node_bitcode))
	{
		search_inds_stack.push(CPUIterTreeNode::getRightChild(search_ind));
		search_codes_stack.push(RIGHT_SEARCH);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::doReportAllNodesDelegation(
											const unsigned char curr_node_bitcode,
											const long long search_ind,
											std::stack<long long> &search_inds_stack,
											std::stack<unsigned char> &search_codes_stack
										)
{
	if (CPUIterTreeNode::hasLeftChild(curr_node_bitcode))
	{
		search_inds_stack.push(CPUIterTreeNode::getLeftChild(search_ind));
		search_codes_stack.push(REPORT_ALL);
	}
	if (CPUIterTreeNode::hasRightChild(curr_node_bitcode))
	{
		search_inds_stack.push(CPUIterTreeNode::getRightChild(search_ind));
		search_codes_stack.push(REPORT_ALL);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <size_t num_T_subarrs>
inline size_t StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::calcTotArrSizeNumTs(const size_t num_elem_slots)
{
	/*
		sizeof(T) >= sizeof(IDType), so alignment requirements for all types satisfied when using maximal compaction

		tot_arr_size_num_Ts = ceil(1/sizeof(T) * num_elem_slots * (sizeof(T) * num_T_subarrs + sizeof(IDType) * num_IDs + 1 B/bitcode * 1 bitcode))
	*/
	// Calculate total size in bytes
	size_t tot_arr_size_bytes = sizeof(T) * num_T_subarrs + 1;
	if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		tot_arr_size_bytes += sizeof(IDType) * num_IDs;
	tot_arr_size_bytes *= num_elem_slots;

	// Divide by sizeof(T)
	size_t tot_arr_size_num_Ts = tot_arr_size_bytes / sizeof(T);
	// If tot_arr_size_bytes % sizeof(T) != 0, then tot_arr_size_num_Ts * sizeof(T) < tot_arr_size_bytes, so add 1 to tot_arr_size_num_Ts
	if (tot_arr_size_bytes % sizeof(T) != 0)
		tot_arr_size_num_Ts++;
	return tot_arr_size_num_Ts;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <size_t num_T_subarrs>
	requires NonVoidType<IDType>
inline size_t StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::calcTotArrSizeNumIDTypes(const size_t num_elem_slots)
{
	/*
		sizeof(IDType) > sizeof(T), so extra padding must be placed before IDType array to ensure alignment requirements are met (hence the distribution of the ceil() function around each addend

		tot_arr_size_num_IDTypes = ceil(1/sizeof(IDType) * num_elem_slots * sizeof(T) * num_T_subarrs)
									+ num_elem_slots * num_IDs
									+ ceil(1/sizeof(IDType) * num_elem_slots * 1 B/bitcode * 1 bitcode)
	*/
	// Calculate size of value arrays in units of number of IDTypes
	const size_t val_arr_size_bytes = num_elem_slots * sizeof(T) * num_T_subarrs;
	const size_t val_arr_size_num_IDTypes = val_arr_size_bytes / sizeof(IDType)
												+ (val_arr_size_bytes % sizeof(IDType) == 0 ? 0 : 1);

	// Calculate size of bitcode array in units of number of IDTypes
	const size_t bitcode_arr_size_bytes = num_elem_slots;
	const size_t bitcode_arr_size_num_IDTypes = bitcode_arr_size_bytes / sizeof(IDType)
												+ (bitcode_arr_size_bytes  % sizeof(IDType) == 0 ? 0 : 1);

	const size_t tot_arr_size_num_IDTypes = val_arr_size_num_IDTypes			// Value array
											+ num_elem_slots * num_IDs			// ID array
											+ bitcode_arr_size_num_IDTypes;		// Bitcode array

	return tot_arr_size_num_IDTypes;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename U>
	requires std::unsigned_integral<U>
U StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::expOfNextGreaterPowerOf2(const U num)
{
	/*
		Smallest power of 2 greater than num is equal to 2^ceil(lg(num + 1))
		ceil(lg(num + 1)) is equal to the number of right bitshifts necessary to make num = 0 (after integer truncation); this method of calcalation is used in order to prevent imprecision of float conversion from causing excessively large (and therefore incorrect) returned integer values
	*/
	unsigned exp = 0;
	while (num >> exp != 0)
		exp++;
	return exp;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
long long StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::binarySearch(
												PointStructTemplate<T, IDType, num_IDs> *const pt_arr,
												size_t *const dim1_val_ind_arr,
												PointStructTemplate<T, IDType, num_IDs> const &elem_to_find,
												const size_t init_ind, const size_t num_elems
											)
{
	size_t low_ind = init_ind;
	size_t high_ind = init_ind + num_elems;
	size_t mid_ind;		// Avoid reinstantiating mid_ind in every iteration
	// Search is done in the range [low_ind, high_ind)
	while (low_ind < high_ind)
	{
		mid_ind = (low_ind + high_ind)/2;
		if (pt_arr[dim1_val_ind_arr[mid_ind]] == elem_to_find
			&& pt_arr[dim1_val_ind_arr[mid_ind]].comparisonTiebreaker(elem_to_find) == 0)
			return mid_ind;
		// elem_to_find is before middle element; recurse on left subarray
		else if (elem_to_find.compareDim1(pt_arr[dim1_val_ind_arr[mid_ind]]) < 0)
			high_ind = mid_ind;
		// elem_to_find is after middle element; recurse on right subarray
		else	// elem_to_find.compareDim1(pt_arr[dim1_val_ind_arr[mid_ind]]) > 0
			low_ind = mid_ind + 1;
	}
	return -1;	// Element not found
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::printRecur(
											std::ostream &os, T *const tree_root,
											const size_t curr_ind, const size_t num_elem_slots,
											std::string prefix, std::string child_prefix
										) const
{
	os << prefix << '(' << getDim1ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< ", " << getDim2ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< "; " << getMedDim1ValsRoot(tree_root, num_elem_slots)[curr_ind];
	if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		os << "; " << getIDsRoot(tree_root, num_elem_slots)[curr_ind];
	os << ')';
	const unsigned char curr_node_bitcode = getBitcodesRoot(tree_root, num_elem_slots)[curr_ind];
	if (CPUIterTreeNode::hasLeftChild(curr_node_bitcode)
			&& CPUIterTreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, CPUIterTreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "├─(R)─ ", child_prefix + "│      ");
		printRecur(os, tree_root, CPUIterTreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
	else if (CPUIterTreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, CPUIterTreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(R)─ ", child_prefix + "       ");
	}
	else if (CPUIterTreeNode::hasLeftChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, CPUIterTreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
}
