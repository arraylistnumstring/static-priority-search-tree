#include "power-of-2-functions.h"

#ifdef DEBUG
#include <iostream>
#endif


// Uses stack instead of dynamic parallelism
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void populateTree(T *const root, const size_t num_elem_slots,
					PointStructTemplate<T, IDType, num_IDs> *const pt_arr,
					size_t *const dim1_val_ind_arr, size_t *dim2_val_ind_arr,
					size_t *dim2_val_ind_arr_secondary,
					const size_t start_ind, const size_t num_elems)
{
	std::stack<size_t> subelems_start_inds;
	std::stack<size_t> num_subelems_stack;
	subelems_start_inds.push(start_ind);
	num_subelems_stack.push(num_elems);

	std::stack<size_t> target_node_inds;
	target_node_inds.push(0);

	// All stacks pop and push simulatenously, so only checking one suffices
	while (!subelems_start_inds.empty())
	{
		size_t left_subarr_num_elems;
		size_t right_subarr_start_ind;
		size_t right_subarr_num_elems;

		size_t *curr_iter_dim2_val_ind_arr;
		size_t *curr_iter_dim2_val_ind_arr_secondary;

		/*
			Because the GPU code won't need to travel back up levels, it is unnecessary to track which array was used at a particular level; however, as the iterative CPU version needs to go to potentially arbitrary levels, it must be able to determine the level in which a node resides to determine which array to use as the primary and which to use as the secondary
			First node, in level 1 (1-indexed), uses the original primary/secondary designations, and subsequent levels alternate
			Must use 1-indexed location as input parameter in order for all nodes of the same level to require the same amount of bitshifting; with this scheme, the first level requires 1 bitshift, the second requires 2, etc. As all odd levels use the original designations of primary and secondary arrays and because calculated output values are either 0 or 1, which translate directly to true (1) or false (0), no equality check is needed.
		*/
		if (expOfMinPowerOf2GreaterThan(target_node_inds.top() + 1) % 2)
		{
			curr_iter_dim2_val_ind_arr = dim2_val_ind_arr;
			curr_iter_dim2_val_ind_arr_secondary = dim2_val_ind_arr_secondary;
		}
		else
		{
			curr_iter_dim2_val_ind_arr = dim2_val_ind_arr_secondary;
			curr_iter_dim2_val_ind_arr_secondary = dim2_val_ind_arr;
		}

		// Find index in dim1_val_ind_arr of PointStruct with maximal dim2_val 
		const long long array_search_res_ind = StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::binarySearch(pt_arr, dim1_val_ind_arr,
																		pt_arr[curr_iter_dim2_val_ind_arr[subelems_start_inds.top()]],
																		subelems_start_inds.top(),
																		num_subelems_stack.top());

		// Check to make sure array_search_res_ind is a valid index (which it should be, since it's an element of pt_arr to begin with)
		assert(array_search_res_ind != -1);

		// Note: potential sign conversion issue when computer memory becomes of size 2^64
		const size_t max_dim2_val_dim1_array_ind = array_search_res_ind;

		StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::constructNode(root, num_elem_slots,
											pt_arr, target_node_inds.top(), num_elems,
											dim1_val_ind_arr, curr_iter_dim2_val_ind_arr,
												curr_iter_dim2_val_ind_arr_secondary,
												max_dim2_val_dim1_array_ind,
											subelems_start_inds, num_subelems_stack,
											left_subarr_num_elems, right_subarr_start_ind,
											right_subarr_num_elems);


		// Track current start index in case current node has left children
		size_t left_subarr_start_ind = subelems_start_inds.top();
		// Pop current node
		subelems_start_inds.pop();
		num_subelems_stack.pop();

		const size_t curr_node_ind = target_node_inds.top();
		target_node_inds.pop();

	#ifdef DEBUG
		/*
			Code that is only compiled when debugging; to define this preprocessor variable, compile with the option -DDEBUG, as in
				nvcc -DDEBUG <source-code-file>

			These variables are explicitly printed because gdb has access to all variables except those under the CPUIterTreeNode namespace (even when explicitly qualified like so); whitespace is meant to clearly separate output of different nodes
		*/
		for (size_t i = 0; i < 10; i++)
			std::cout << '\n';

		const unsigned char curr_node_bitcode = StaticPSTCPUIter<T, PointStructTemplate, IDType, num_IDs>::getBitcodesRoot(root, num_elem_slots)[curr_node_ind];
		std::cout << "Current node has left child: " << CPUIterTreeNode::hasLeftChild(curr_node_bitcode) << '\n';
		std::cout << "Current node has right child: " << CPUIterTreeNode::hasRightChild(curr_node_bitcode) << '\n';
		std::cout << "Current node has children: " << CPUIterTreeNode::hasChildren(curr_node_bitcode) << '\n';
	#endif

		// Update information for next iteration
		if (right_subarr_num_elems > 0)
		{
			target_node_inds.push(CPUIterTreeNode::getRightChild(curr_node_ind));
			subelems_start_inds.push(right_subarr_start_ind);
			num_subelems_stack.push(right_subarr_num_elems);
		}
		if (left_subarr_num_elems > 0)
		{
			target_node_inds.push(CPUIterTreeNode::getLeftChild(curr_node_ind));
			subelems_start_inds.push(left_subarr_start_ind);
			num_subelems_stack.push(left_subarr_num_elems);
		}

		// At this point in the GPU code, every thread must swap its primary and secondary dim2_val_ind_arr pointers because between levels, the source and target index arrays that are ordered by dimension 2 swap
	}
}
