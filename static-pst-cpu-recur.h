#ifndef STATIC_PST_CPU_RECUR_H
#define STATIC_PST_CPU_RECUR_H

#include "static-priority-search-tree.h"

template <typename T>
// public superclass means that all public and protected members of base-class retain their access status in the subclass
class StaticPSTCPURecur : public StaticPrioritySearchTree<T>
{
	// Throws a compile-time error if T is not of arithmetic (numeric) type
	// static_assert() and std::is_arithmetic are C++11 features
	// static_assert() must have two arguments to compile on CIMS
	static_assert(std::is_arithmetic<T>::value, "Input type T not of arithmetic type");

	public:
		StaticPSTCPURecur(PointStruct<T> *pt_arr, size_t num_elems);
		virtual ~StaticPSTCPURecur()
		{
			if (root != nullptr)
				delete[] root
		};

		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const;

		// Initial input value for num_res_elems is the array initialisation size
		virtual PointStruct<T>* threeSidedSearch(size_t &num_res_elems, T min_dim1_val, T max_dim1_val, T min_dim2_val)
		{
			if (root == nullptr)
			{
				std::cout << "Tree is empty; nothing to search\n";
				return nullptr;
			}
			size_t pt_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			PointStruct<T>* pt_arr = new PointStruct<T>[pt_arr_size];
			num_res_elems = 0;
			threeSidedSearchRecur(pt_arr, num_res_elems, pt_arr_size, *root, min_dim1_val, max_dim1_val, min_dim2_val);

			// Ensure that no more memory is taken up than needed
			if (pt_arr_size > num_res_elems)
				PointStruct<T>::resizePointStructArray(pt_arr, pt_arr_size, num_res_elems);

			return pt_arr;
		};
		virtual PointStruct<T>* twoSidedLeftSearch(size_t &num_res_elems, T max_dim1_val, T min_dim2_val)
		{
			if (root == nullptr)
			{
				std::cout << "Tree is empty; nothing to search\n";
				return nullptr;
			}
			size_t pt_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			PointStruct<T>* pt_arr = new PointStruct<T>[pt_arr_size];
			num_res_elems = 0;
			twoSidedLeftSearchRecur(pt_arr, num_res_elems, pt_arr_size, *root, max_dim1_val, min_dim2_val);

			// Ensure that no more memory is taken up than needed
			if (pt_arr_size > num_res_elems)
				PointStruct<T>::resizePointStructArray(pt_arr, pt_arr_size, num_res_elems);

			return pt_arr;
		};
		virtual PointStruct<T>* twoSidedRightSearch(size_t &num_res_elems, T min_dim1_val, T min_dim2_val)
		{
			if (root == nullptr)
			{
				std::cout << "Tree is empty; nothing to search\n";
				return nullptr;
			}
			size_t pt_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			PointStruct<T>* pt_arr = new PointStruct<T>[pt_arr_size];
			num_res_elems = 0;
			twoSidedRightSearchRecur(pt_arr, num_res_elems, pt_arr_size, *root, min_dim1_val, min_dim2_val);

			// Ensure that no more memory is taken up than needed
			if (pt_arr_size > num_res_elems)
				PointStruct<T>::resizePointStructArray(pt_arr, pt_arr_size, num_res_elems);

			return pt_arr;
		};

	private:
		// Want unique copies of each tree, so no assignment or copying allowed
		StaticPSTCPURecur& operator=(StaticPSTCPURecur &tree);	// assignment operator
		StaticPSTCPURecur(StaticPSTCPURecur &tree);	// copy constructor

		// Declare nested class and define in implementation file
		class TreeNode;

		TreeNode *root;

		// Recursive constructor helper to populate the tree
		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		// Note that **const ensures that a second-level pointer (**) is constant, but that the pointer to which it points (i.e. a first-level pointer, *) is not
		void populateTreeRecur(TreeNode &subtree_root, PointStruct<T> *const *const dim1_val_ptr_subarr, PointStruct<T> *const *const dim2_val_ptr_subarr, const size_t num_elems);
		// Returns index in dim1_val_ptr_arr of elem_to_find
		size_t binarySearch(PointStruct<T> *const *const dim1_val_ptr_arr, PointStruct<T> &elem_to_find, const size_t num_elems);

		void printRecur(std::ostream &os, const TreeNode &subtree_root, std::string prefix, std::string child_prefix) const;

		// Search-related helper functions
		void threeSidedSearchRecur(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T min_dim1_val, T max_dim1_val, T min_dim2_val);
		void twoSidedLeftSearchRecur(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T max_dim1_val, T min_dim2_val);
		void twoSidedRightSearchRecur(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T min_dim1_val, T min_dim2_val);
		void reportAllNodes(PointStruct<T> *&pt_arr, size_t &num_res_elems, size_t &pt_arr_size, TreeNode &subtree_root, T min_dim2_val);

	// Allow printing operator << to be declared for TreeNode
	// For friend functions of template classes, for the compiler to recognise the function as a template function, it is necessary to either pre-declare each template friend function before the template class and modify the class-internal function declaration with an additional <> between the operator and the parameter list; or to simply define the friend function when it is declared
	//	https://isocpp.org/wiki/faq/templates#template-friends
	friend std::ostream &operator<<(std::ostream &os, const TreeNode &tn)
	{
		tn.print(os);
		return os;
	};
};

// Implementation file; for class templates, implementations must be in the same file as the declaration so that the compiler can access them
#include "static-pst-cpu-recur-tree-node.tpp"
#include "static-pst-cpu-recur.tpp"

#endif
