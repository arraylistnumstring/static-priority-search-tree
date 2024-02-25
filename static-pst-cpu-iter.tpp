#include <algorithm>	// To use sort()
#include <cstring>		// To use memcpy()
#include <string>		// To use string-building functions

#include "err-chk.h"

template <typename T>
StaticPSTCPUIter<T>::StaticPSTCPUIter(PointStructCPUIter<T> *const &pt_arr, size_t num_elems)
{
	if (num_elems == 0)
	{
		root = nullptr;
		num_elem_slots = 0;
		return;
	}

	this->num_elems - num_elems;
	// Number of element slots in each container subarray is nextGreaterPowerOf2(num_elems) - 1
	num_elem_slots = nextGreaterPowerOf2(num_elems) - 1;

	// Calculate total array size in units of sizeof(T) bytes so that array will satisfy datatype T's alignment requirements (as T is the largest datatype present in this data structure)
	size_t tot_arr_size_num_Ts = calcTotArrSizeNumTs(num_elem_slots);

	// Allocate as a T array so that alignment requirements for larger data types are obeyed
	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	root = new T[tot_arr_size_num_Ts]();
	if (root == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type " + typeid(T).name() + " to root");

	// Create two arrays of PointStructCPUIter indices for processing PointStructCPUIter objects
	size_t *dim1_val_ind_arr = new size_t[num_elems];
	if (dim1_val_ind_arr == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type size_t to dim1_val_ind_arr");
	size_t *dim2_val_ind_arr = new size_t[num_elems];
	if (dim2_val_ind_arr == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type size_t to dim2_val_ind_arr");
	size_t *dim2_val_ind_arr_secondary = new size_t[num_elems];
	if (dim2_val_ind_arr_secondary == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems) + " elements of type size_t to dim2_val_ind_arr_secondary");

	for (size_t i = 0; i < num_elems; i++)
		dim1_val_ind_arr[i] = dim2_val_ind_arr[i] = i;

	// Sort dimension-1 values index array in ascending order; in-place sort using a curried comparison function
	std::sort(dim1_val_ind_arr, dim1_val_ind_arr + num_elems,
				[](PointStructCPUIter<T> *const &pt_arr)
					{
						// [&] captures all variables in enclosing scope by reference so that they can be used within the body of the lambda function
						return [&](const size_t &i, const size_t &j)
							{
								return pt_arr[i].compareDim1(pt_arr[j]) < 0;
							};
					}(pt_arr));	// Parentheses immediately after a lambda definition serves to call it with the given parameter

	// Sort dimension-2 values index array in descending order; in-place sort using a curried comparison function
	std::sort(dim2_val_ind_arr, dim2_val_ind_arr + num_elems,
				[](PointStructCPUIter<T> *const &pt_arr)
					{
						return [&](const size_t &i, const size_t &j)
							{
								return pt_arr[i].compareDim2(pt_arr[j]) > 0;
							};
					}(pt_arr));


#ifdef DEBUG
	/*
		Code that is only compiled when debugging; to define this preprocessor variable, compile with the option -DDEBUG, as in
			nvcc -DDEBUG <source-code-file>

		Explicitly printed to sanity-check the corresponding code in StaticPSTGPU
	*/
	std::cout << "Would call populateTree() with " << 1 << " block, " << nextGreaterPowerOf2(32 - 1) << " threads, " << (nextGreaterPowerOf2(32 - 1) * sizeof(size_t) << 1) << " B of shared memory\n";
#endif
	populateTree(root, num_elem_slots, pt_arr, dim1_val_ind_arr, dim2_val_ind_arr, dim2_val_ind_arr_secondary, 0, num_elems);

	delete[] dim1_val_ind_arr;
	delete[] dim2_val_ind_arr;
}

// const keyword after method name indicates that the method does not modify any data members of the associated class
template <typename T>
void StaticPSTCPUIter<T>::print(std::ostream &os) const
{
	if (num_elem_slots == 0)
	{
		os << "Tree is empty\n";
		return;
	}

	std::string prefix = "";
	std::string child_prefix = "";
	printRecur(os, root, 0, num_elem_slots, prefix, child_prefix);
}

template <typename T>
PointStructCPUIter<T>* StaticPSTCPUIter<T>::threeSidedSearch(size_t &num_res_elems, T min_dim1_val, T max_dim1_val, T min_dim2_val)
{
	if (num_elems == 0)
	{
		std::cout << "Tree is empty; nothing to search\n";
		num_res_elems = 0;
		return nullptr;
	}

	size_t res_pt_arr_size = num_elems;
	PointStructCPUIter<T>* res_pt_arr = new PointStructCPUIter<T>[res_pt_arr_size];
	num_res_elems = 0;
	// TODO: search

	// Ensure that no more memory is taken up than needed
	if (res_pt_arr_size > num_res_elems)
		PointStruct<T>::resizePointStructArray(res_pt_arr, res_pt_arr_size, num_res_elems);

	return res_pt_arr;
}

template <typename T>
PointStructCPUIter<T>* StaticPSTCPUIter<T>::twoSidedLeftSearch(size_t &num_res_elems, T max_dim1_val, T min_dim2_val)
{
	if (num_elems == 0)
	{
		std::cout << "Tree is empty; nothing to search\n";
		num_res_elems = 0;
		return nullptr;
	}

	size_t res_pt_arr_size = num_elems;
	PointStructCPUIter<T>* res_pt_arr = new PointStructCPUIter<T>[res_pt_arr_size];
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
				res_pt_arr[num_res_elems].dim1_val = curr_node_dim1_val;
				res_pt_arr[num_res_elems].dim2_val = curr_node_dim2_val;
				num_res_elems++;
			}

			// Delegation/further activity down this branch necessary only if this node has children and can therefore be searched
			if (TreeNode::hasChildren(curr_node_bitcode))
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
	if (res_pt_arr_size > num_res_elems)
		PointStruct<T>::resizePointStructArray(res_pt_arr, res_pt_arr_size, num_res_elems);

	return res_pt_arr;
}

template <typename T>
PointStructCPUIter<T>* StaticPSTCPUIter<T>::twoSidedRightSearch(size_t &num_res_elems, T min_dim1_val, T min_dim2_val)
{
	if (num_elems == 0)
	{
		std::cout << "Tree is empty; nothing to search\n";
		num_res_elems = 0;
		return nullptr;
	}

	size_t res_pt_arr_size = num_elems;
	PointStructCPUIter<T>* res_pt_arr = new PointStructCPUIter<T>[res_pt_arr_size];
	num_res_elems = 0;
	// TODO: search

	// Ensure that no more memory is taken up than needed
	if (res_pt_arr_size > num_res_elems)
		PointStruct<T>::resizePointStructArray(res_pt_arr, res_pt_arr_size, num_res_elems);

	return res_pt_arr;
}

// static keyword should only be used when declaring a function in the header file
template <typename T>
void StaticPSTCPUIter<T>::constructNode(T *const &root,
										const size_t &num_elem_slots,
										PointStructCPUIter<T> *const &pt_arr,
										size_t &target_node_ind,
										const size_t &num_elems,
										size_t *const &dim1_val_ind_arr,
										size_t *&dim2_val_ind_arr,
										size_t *&dim2_val_ind_arr_secondary,
										const size_t &max_dim2_val_dim1_array_ind,
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
		TreeNode::setLeftChild(getBitcodesRoot(root, num_elem_slots), target_node_ind);

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
		TreeNode::setRightChild(getBitcodesRoot(root, num_elem_slots), target_node_ind);

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
	if (TreeNode::hasChildren(getBitcodesRoot(root, num_elem_slots)[target_node_ind]))
	{
		size_t left_dim2_subarr_iter_ind = subelems_start_inds.top();
		size_t right_dim2_subarr_iter_ind = right_subarr_start_ind;

		// Skip over first (largest) element in dim2_val_ind_arr, as it has already been placed in the current node
		for (size_t i = subelems_start_inds.top() + 1;
				i < subelems_start_inds.top() + num_subelems_stack.top();
				i++)
		{
			// dim2_val_ind_arr[i] is the index of a PointStructCPUIter that comes before or is the PointStructCPUIter of median dim1 value in dim1_val_ind_arr
			if (pt_arr[dim2_val_ind_arr[i]].compareDim1(pt_arr[dim1_val_ind_arr[median_dim1_val_ind]]) <= 0)
				// Postfix ++ returns the current value before incrementing
				dim2_val_ind_arr_secondary[left_dim2_subarr_iter_ind++] = dim2_val_ind_arr[i];
			// dim2_val_ind_arr[i] is the index of a PointStructCPUIter that comes after the PointStructCPUIter of median dim1 value in dim1_val_ind_arr
			else
				dim2_val_ind_arr_secondary[right_dim2_subarr_iter_ind++] = dim2_val_ind_arr[i];
		}
	}
}

template <typename T>
size_t StaticPSTCPUIter<T>::calcTotArrSizeNumTs(const size_t num_elem_slots)
{
	/*
		tot_arr_size_num_Ts = ceil(num_elem_slots * (num_val_subarrs + num_Ts/bitcode))
							= ceil(num_elem_slots * (num_val_subarrs + 1 B/bitcode * 1 T/#Bs))
							= ceil(num_elem_slots * (num_val_subarrs + 1/sizeof(T)))
							= ceil(num_elem_slots * num_val_subarrs + num_elem_slots / sizeof(T))
							= num_elem_slots * num_val_subarrs + ceil(num_elem_slots / sizeof(T))
			With integer truncation:
				if num_elem_slots % codes_per_byte != 0:
							= num_elem_slots * num_val_subarrs + num_elem_slots / sizeof(T) + 1
				if num_elem_slots % codes_per_byte == 0:
							= num_elem_slots * num_val_subarrs + num_elem_slots / sizeof(T)
	*/
	size_t tot_arr_size_num_Ts = num_val_subarrs * num_elem_slots + num_elem_slots/sizeof(T);
	if (num_elem_slots % sizeof(T) != 0)
		tot_arr_size_num_Ts++;
	return tot_arr_size_num_Ts;
}

template <typename T>
size_t StaticPSTCPUIter<T>::expOfNextGreaterPowerOf2(const size_t num)
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

template <typename T>
long long StaticPSTCPUIter<T>::binarySearch(PointStructCPUIter<T> *const &pt_arr, size_t *const &dim1_val_ind_arr, PointStructCPUIter<T> &elem_to_find, const size_t &init_ind, const size_t &num_elems)
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

template <typename T>
void StaticPSTCPUIter<T>::printRecur(std::ostream &os, T *const &tree_root, const size_t curr_ind, const size_t num_elem_slots, std::string prefix, std::string child_prefix) const
{
	os << prefix << '(' << getDim1ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< ", " << getDim2ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< "; " << getMedDim1ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< ')';
	const unsigned char curr_node_bitcode = getBitcodesRoot(tree_root, num_elem_slots)[curr_ind];
	if (TreeNode::hasLeftChild(curr_node_bitcode)
			&& TreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, TreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "├─(R)─ ", child_prefix + "│      ");
		printRecur(os, tree_root, TreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
	else if (TreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, TreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(R)─ ", child_prefix + "       ");
	}
	else if (TreeNode::hasLeftChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, TreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
}
