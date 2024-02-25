#include <algorithm>	// To use sort()
#include <cstring>		// To use memcpy()
#include <string>		// To use string-building functions

#include "print-array.h"

template <typename T>
StaticPSTCPURecur<T>::StaticPSTCPURecur(PointStruct<T> *pt_arr, size_t num_elems)
{
	if (num_elems == 0)
	{
		root = nullptr;
		return;
	}

	PointStruct<T> **dim1_val_ptr_arr = new PointStruct<T>*[num_elems]();
	PointStruct<T> **dim2_val_ptr_arr = new PointStruct<T>*[num_elems]();

	for (size_t i = 0; i < num_elems; i++)
		dim1_val_ptr_arr[i] = dim2_val_ptr_arr[i] = pt_arr + i;

	// Sort dimension-1 values pointer array in ascending order; in-place sort
	std::sort(dim1_val_ptr_arr, dim1_val_ptr_arr + num_elems,
				[](PointStruct<T> *const &node_ptr_1, PointStruct<T> *const &node_ptr_2)
				{
					return node_ptr_1->compareDim1(*node_ptr_2) < 0;
				});

	// Sort dim2_val pointer array in descending order; in-place sort
	std::sort(dim2_val_ptr_arr, dim2_val_ptr_arr + num_elems,
				[](PointStruct<T> *const &node_ptr_1, PointStruct<T> *const &node_ptr_2)
				{
					return node_ptr_1->compareDim2(*node_ptr_2) > 0;
				});

#ifdef DEBUG
	/*
		Code that is only compiled when debugging; to define this preprocessor variable, compile with the option -DDEBUG, as in
			nvcc -DDEBUG <source-code-file>

		Explicitly printed to sanity-check the corresponding code in StaticPSTCPUIter
	*/
	std::cout << "PointStructs ordered by dimension 1 via pointers:\n";
	printArrayDereferenced(dim1_val_ptr_arr, 0, num_elems);
	std::cout << "\tCorresponding indices:\n\t\t";
	printArrayOffsetFromStart(pt_arr, dim1_val_ptr_arr, 0, num_elems);
	std::cout << "PointStructs ordered by dimension 2 via pointers:\n";
	printArrayDereferenced(dim2_val_ptr_arr, 0, num_elems);
	std::cout << "\tCorresponding indices:\n\t\t";
	printArrayOffsetFromStart(pt_arr, dim2_val_ptr_arr, 0, num_elems);
#endif

	// Minimum number of array slots necessary to construct tree given it is fully balanced by construction and given the unknown placement of nodes in the partially empty last row
	// Number of slots in container array is 2^ceil(lg(num_elems + 1)) - 1
	// ceil(lg(num_elems + 1)) is equal to the number of right bitshifts necessary to make num_elems = 0 (after integer truncation); this method of calcalation is used in order to prevent errors in precision of float conversion from causing an unnecessarily large array to be assigned
	unsigned exp = 0;
	while (num_elems >> exp != 0)
		exp++;

	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	root = new TreeNode[(1 << exp) - 1]();

	populateTreeRecur(*(root), dim1_val_ptr_arr, dim2_val_ptr_arr, num_elems);
}

template <typename T>
StaticPSTCPURecur<T>::~StaticPSTCPURecur()
{
	delete[] root;
}

template <typename T>
void StaticPSTCPURecur<T>::populateTreeRecur(TreeNode &subtree_root, PointStruct<T> *const *const dim1_val_ptr_subarr, PointStruct<T> *const *const dim2_val_ptr_subarr, const size_t num_elems)
{
	// Find index in dim1_val_ptr_subarr of PointStruct with maximal dim2_val 
	size_t max_dim2_val_dim1_array_ind = binarySearch(dim1_val_ptr_subarr, *dim2_val_ptr_subarr[0], num_elems);

	size_t median_dim1_val_ind;
	size_t left_subarr_num_elems;
	size_t right_subarr_num_elems;

	PointStruct<T> **left_dim1_val_ptr_subarr;
	PointStruct<T> **right_dim1_val_ptr_subarr;

	PointStruct<T> **left_dim2_val_ptr_subarr;
	PointStruct<T> **right_dim2_val_ptr_subarr;


	// Treat *dim1_val_ptr_subarr[max_dim2_val_dim1_array_ind] as a removed element (but don't actually remove the element for performance reasons
	if (num_elems == 1)		// Base case
	{
		median_dim1_val_ind = 0;
		left_subarr_num_elems = 0;
		right_subarr_num_elems = 0;
	}
	// max_dim2_val originally comes from the part of the array to the left of median_dim1_val
	else if (max_dim2_val_dim1_array_ind < num_elems/2)
	{
		// As median values are always put in the left subtree, when the subroot value comes from the left subarray, the median index is given by num_elems/2, which evenly splits the array if there are an even number of elements remaining or makes the left subtree larger by one element if there are an odd number of elements remaining
		median_dim1_val_ind = num_elems/2;
		// max_dim2_val has been removed from the left subarray, so there are median_dim1_val_ind elements remaining on the left side
		left_subarr_num_elems = median_dim1_val_ind;
		right_subarr_num_elems = num_elems - median_dim1_val_ind - 1;
	}
	// max_dim2_val originally comes from the part of the array to the right of median_dim1_val
	else	// max_dim2_val_dim1_array_ind >= num_elems/2
	{
		median_dim1_val_ind = num_elems/2 - 1;

		left_subarr_num_elems = median_dim1_val_ind + 1;
		right_subarr_num_elems = num_elems - median_dim1_val_ind - 2;
	}


	// Set current node data
	subtree_root.setTreeNode(*dim2_val_ptr_subarr[0],
							 dim1_val_ptr_subarr[median_dim1_val_ind]->dim1_val);


	if (left_subarr_num_elems > 0)
	{
		subtree_root.setLeftChild();

		left_dim1_val_ptr_subarr = new PointStruct<T>*[left_subarr_num_elems];
		left_dim2_val_ptr_subarr = new PointStruct<T>*[left_subarr_num_elems];

		// Always place median value in left subtree
		// max_dim2_val is to the left of median_dim1_val in dim1_val_ptr_subarr; do two-piece left copy, skipping max_dim2_val
		if (max_dim2_val_dim1_array_ind < median_dim1_val_ind)
		{
			std::memcpy(left_dim1_val_ptr_subarr,
						dim1_val_ptr_subarr,
						max_dim2_val_dim1_array_ind * sizeof(PointStruct<T>*));
			std::memcpy(left_dim1_val_ptr_subarr + max_dim2_val_dim1_array_ind,
						dim1_val_ptr_subarr + max_dim2_val_dim1_array_ind + 1,
						(median_dim1_val_ind - max_dim2_val_dim1_array_ind) * sizeof(PointStruct<T>*));
		}
		// max_dim2_val is to the right of median_dim1_val_ind in dim1_val_ptr_subarr; do one-piece left copy
		else	// max_dim2_val_dim1_array_ind > median_dim1_val_ind; the two values are never equal
			std::memcpy(left_dim1_val_ptr_subarr,
						dim1_val_ptr_subarr,
						(median_dim1_val_ind + 1) * sizeof(PointStruct<T>*));
	}
	if (right_subarr_num_elems > 0)
	{
		subtree_root.setRightChild();

		right_dim1_val_ptr_subarr = new PointStruct<T>*[right_subarr_num_elems];
		right_dim2_val_ptr_subarr = new PointStruct<T>*[right_subarr_num_elems];

		// max_dim2_val is to the left of median_dim1_val in dim1_val_ptr_subarr; do one-piece right copy
		if (max_dim2_val_dim1_array_ind < median_dim1_val_ind)
			std::memcpy(right_dim1_val_ptr_subarr,
						dim1_val_ptr_subarr + median_dim1_val_ind + 1,
						right_subarr_num_elems * sizeof(PointStruct<T>*));
		// max_dim2_val is to the right of median_dim1_val_ind in dim1_val_ptr_subarr; do two-piece right copy, skipping max_dim2_val
		else	// max_dim2_val_dim1_array_ind > median_dim1_val_ind; the two values are never equal
		{
			std::memcpy(right_dim1_val_ptr_subarr,
						dim1_val_ptr_subarr + median_dim1_val_ind + 1,
						(max_dim2_val_dim1_array_ind - median_dim1_val_ind - 1) * sizeof(PointStruct<T>*));
			std::memcpy(right_dim1_val_ptr_subarr + max_dim2_val_dim1_array_ind - median_dim1_val_ind - 1,
						dim1_val_ptr_subarr + max_dim2_val_dim1_array_ind + 1,
						(num_elems - max_dim2_val_dim1_array_ind - 1) * sizeof(PointStruct<T>*));
		}
	}


	// Iterate through dim2_val_ptr_subarr, placing data with lower dimension-1 value in the left subtree and data with higher dimension-1 value in the right subtree
	// Note that because subarrays' sizes were allocated based on this same data, there should not be a need to check that left_dim2_subarr_iter_ind < left_subarr_num_elems (and similarly for the right side)
	if (subtree_root.hasChildren())
	{
		size_t left_dim2_subarr_iter_ind = 0;
		size_t right_dim2_subarr_iter_ind = 0;

		for (size_t i = 1; i < num_elems; i++)
		{
			// dim2_val_ptr_subarr[i] comes before or is the median value in dim1_val_ptr_subarr
			if (dim2_val_ptr_subarr[i]->compareDim1(*dim1_val_ptr_subarr[median_dim1_val_ind]) <= 0)
				// Postfix ++ returns the current value before incrementing
				left_dim2_val_ptr_subarr[left_dim2_subarr_iter_ind++] = dim2_val_ptr_subarr[i];
			// dim2_val_ptr_subarr[i] comes after the median value in dim1_val_ptr_subarr
			else
				right_dim2_val_ptr_subarr[right_dim2_subarr_iter_ind++] = dim2_val_ptr_subarr[i];
		}
	}

	delete[] dim1_val_ptr_subarr;
	delete[] dim2_val_ptr_subarr;

	// Last statements in the function, despite the possibility of nesting them in the earlier checks for left_subarr_num_elems > 0 and right_subarr_num_elems > 0, in order to be as close to tail recursion as possible and minimise stack frame size
	// subtree_root.hasLeftChild() and subtree_root.hasRightChild() methods used to take advantage of bitwise operations being faster than addition and subtraction (as would be used with a > 0 comparison)
	if (subtree_root.hasLeftChild())
		populateTreeRecur(subtree_root.getLeftChild(root), left_dim1_val_ptr_subarr, left_dim2_val_ptr_subarr, left_subarr_num_elems);
	if (subtree_root.hasRightChild())
		populateTreeRecur(subtree_root.getRightChild(root), right_dim1_val_ptr_subarr, right_dim2_val_ptr_subarr, right_subarr_num_elems);
}

template <typename T>
size_t StaticPSTCPURecur<T>::binarySearch(PointStruct<T> *const *const dim1_val_ptr_arr, PointStruct<T> &elem_to_find, const size_t num_elems)
{
	size_t low_ind = 0;
	size_t high_ind = num_elems;
	// Search is done in the range [low_ind, high_ind)
	while (low_ind < high_ind)
	{
		size_t mid_ind = (low_ind + high_ind)/2;
		// Location in dim1_val_ptr_arr of elem_to_find has been found
		if (*dim1_val_ptr_arr[mid_ind] == elem_to_find && dim1_val_ptr_arr[mid_ind]->comparisonTiebreaker(elem_to_find) == 0)
			return mid_ind;
		// elem_to_find is before middle element; recurse on left subarray
		else if (elem_to_find.compareDim1(*dim1_val_ptr_arr[mid_ind]) < 0)
			high_ind = mid_ind;
		// elem_to_find is after middle element; recurse on right subarray
		else	// elem_to_find.compareDim1(*dim1_val_ptr_arr[mid_ind]) > 0
			low_ind = mid_ind + 1;
	}
	return -1;	// Element not found
}

// const keyword after method name indicates that the method does not modify any data members of the associated class
template <typename T>
void StaticPSTCPURecur<T>::print(std::ostream &os) const
{
	if (root == nullptr)
	{
		os << "Tree is empty\n";
		return;
	}
	std::string prefix = "";
	std::string child_prefix = "";
	printRecur(os, *root, prefix, child_prefix);
}

template <typename T>
void StaticPSTCPURecur<T>::printRecur(std::ostream &os, const TreeNode &subtree_root, std::string prefix, std::string child_prefix) const
{
	os << prefix << subtree_root;
	if (subtree_root.hasLeftChild() && subtree_root.hasRightChild())
	{
		printRecur(os, subtree_root.getRightChild(root), '\n' + child_prefix + "├─(R)─ ", child_prefix + "│      ");
		printRecur(os, subtree_root.getLeftChild(root), '\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
	else if (subtree_root.hasRightChild())
	{
		printRecur(os, subtree_root.getRightChild(root), '\n' + child_prefix + "└─(R)─ ", child_prefix + "       ");
	}
	else if (subtree_root.hasLeftChild())
	{
		printRecur(os, subtree_root.getLeftChild(root), '\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
}

template <typename T>
void StaticPSTCPURecur<T>::reportAllNodes(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T min_dim2_val)
{
	if (min_dim2_val > subtree_root.pt.dim2_val) return;	// No more nodes to report

	// Allow polymorphism to determine which type of PointStruct needs to be instantiated to capture all relevant information
	pt_arr[num_res_elems++] = subtree_root.pt;
	// Dynamically resize array
	if (num_res_elems == pt_arr_size)
		PointStruct<T>::resizePointStructArray(pt_arr, pt_arr_size, pt_arr_size << 1);

	if (subtree_root.hasLeftChild())
		reportAllNodes(pt_arr, num_res_elems, pt_arr_size, subtree_root.getLeftChild(root), min_dim2_val);
	if (subtree_root.hasRightChild())
		reportAllNodes(pt_arr, num_res_elems, pt_arr_size, subtree_root.getRightChild(root), min_dim2_val);
}

template <typename T>
void StaticPSTCPURecur<T>::threeSidedSearchRecur(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T min_dim1_val, T max_dim1_val, T min_dim2_val)
{
	if (min_dim2_val > subtree_root.pt.dim2_val) return;	// No more nodes to report

	// Check if this node satisfies the search criteria
	else if (min_dim1_val <= subtree_root.pt.dim1_val && subtree_root.pt.dim1_val <= max_dim1_val)
	{
		// Allow polymorphism to determine which type of PointStruct needs to be instantiated to capture all relevant information
		pt_arr[num_res_elems++] = subtree_root.pt;
		// Dynamically resize array
		if (num_res_elems == pt_arr_size)
			PointStruct<T>::resizePointStructArray(pt_arr, pt_arr_size, pt_arr_size << 1);
	}

	// Continue search

	// Search interval is fully to the right of median_dim1_val
	if (subtree_root.median_dim1_val < min_dim1_val && subtree_root.hasRightChild())
		threeSidedSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getRightChild(root), min_dim1_val, max_dim1_val, min_dim2_val);
	// Search interval is fully to the left of median_dim1_val
	else if (subtree_root.median_dim1_val > max_dim1_val && subtree_root.hasLeftChild())
		threeSidedSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getLeftChild(root), min_dim1_val, max_dim1_val, min_dim2_val);
	// Max value of search interval is bounded by median_dim1_val, so can do a two-sided search on left subtree
	else
	{
		// median_dim1_val is in the boundary of the dimension-1 value interval [min_dim1_val, max_dim1_val]; split into 2 two-sided searches; the upper bound is typically open, but if there are duplicates and one copy of a median point happens to fall into either subtree, both trees must be traversed for correctness
		if (subtree_root.hasLeftChild())
			twoSidedRightSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getLeftChild(root), min_dim1_val, min_dim2_val);
		if (subtree_root.hasRightChild())
			twoSidedLeftSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getRightChild(root), max_dim1_val, min_dim2_val);
	}
}

template <typename T>
void StaticPSTCPURecur<T>::twoSidedLeftSearchRecur(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T max_dim1_val, T min_dim2_val)
{
	if (min_dim2_val > subtree_root.pt.dim2_val) return;	// No more nodes to report

	// Check if this node satisfies the search criteria
	if (subtree_root.pt.dim1_val <= max_dim1_val)
	{
		// Allow polymorphism to determine which type of PointStruct needs to be instantiated to capture all relevant information
		pt_arr[num_res_elems++] = subtree_root.pt;
		// Dynamically resize array
		if (num_res_elems == pt_arr_size)
			PointStruct<T>::resizePointStructArray(pt_arr, pt_arr_size, pt_arr_size << 1);
	}

	// Continue search

	// Max boundary is to right of median_dim1_val, so all left subtree dimension-1 values are valid, and potentially some in the right subtree
	if (subtree_root.median_dim1_val < max_dim1_val)
	{
		if (subtree_root.hasLeftChild())
			// Report all nodes in left subtree with dim2_val higher than min_dim2_val
			reportAllNodes(pt_arr, num_res_elems, pt_arr_size, subtree_root.getLeftChild(root), min_dim2_val);
		if (subtree_root.hasRightChild())
			twoSidedLeftSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getRightChild(root), max_dim1_val, min_dim2_val);
	}
	// Only left subtree dimension-1 values can be valid
	else if (subtree_root.hasLeftChild())
		twoSidedLeftSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getLeftChild(root), max_dim1_val, min_dim2_val);
}

template <typename T>
void StaticPSTCPURecur<T>::twoSidedRightSearchRecur(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T min_dim1_val, T min_dim2_val)
{
	if (min_dim2_val > subtree_root.pt.dim2_val) return;	// No more nodes to report

	// Check if this node satisfies the search criteria
	if (subtree_root.pt.dim1_val >= min_dim1_val)
	{
		// Allow polymorphism to determine which type of PointStruct needs to be instantiated to capture all relevant information
		pt_arr[num_res_elems++] = subtree_root.pt;
		// Dynamically resize array
		if (num_res_elems == pt_arr_size)
			PointStruct<T>::resizePointStructArray(pt_arr, pt_arr_size, pt_arr_size << 1);
	}

	// Continue search

	// Min boundary is to left of median_dim1_val or contains median_dim1_val, so all right subtree dimension-1 values are valid, and potentially some in the left subtree
	if (subtree_root.median_dim1_val >= min_dim1_val)
	{
		if (subtree_root.hasRightChild())
			// Report all nodes in right subtree with dim2_val higher than min_dim2_val
			reportAllNodes(pt_arr, num_res_elems, pt_arr_size, subtree_root.getRightChild(root), min_dim2_val);
		if (subtree_root.hasLeftChild())
			twoSidedRightSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getLeftChild(root), min_dim1_val, min_dim2_val);
	}
	// Only right subtree dimension-1 values can be valid
	else if (subtree_root.hasRightChild())
		twoSidedRightSearchRecur(pt_arr, num_res_elems, pt_arr_size, subtree_root.getRightChild(root), min_dim1_val, min_dim2_val);
}
