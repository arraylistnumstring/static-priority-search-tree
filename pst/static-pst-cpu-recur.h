#ifndef STATIC_PST_CPU_RECUR_H
#define STATIC_PST_CPU_RECUR_H

#include "cpu-recur-tree-node.h"
#include "resize-array.h"
#include "static-priority-search-tree.h"

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType=void, size_t num_IDs=0>
// public superclass means that all public and protected members of base-class retain their access status in the subclass
class StaticPSTCPURecur : public StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_IDs>
{
	public:
		StaticPSTCPURecur(PointStructTemplate<T, IDType, num_IDs> *pt_arr, size_t num_elems);
		virtual ~StaticPSTCPURecur()
		{
			if (root != nullptr)
				delete[] root;
		};

		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const;

		// Initial input value for num_res_elems is the array initialisation size
		template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void threeSidedSearch(size_t &num_res_elems, RetType *&res_arr, T min_dim1_val, T max_dim1_val, T min_dim2_val)
		{
			if (root == nullptr)
			{
				std::cout << "Tree is empty; nothing to search\n";
				res_arr = nullptr;
				return;
			}
			size_t res_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			res_arr = new RetType[res_arr_size];
			num_res_elems = 0;
			threeSidedSearchRecur(res_arr, num_res_elems, res_arr_size, *root, min_dim1_val, max_dim1_val, min_dim2_val);

			// Ensure that no more memory is taken up than needed
			if (res_arr_size > num_res_elems)
				resizeArray(res_arr, res_arr_size, num_res_elems);
		};
		template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedLeftSearch(size_t &num_res_elems, RetType *&res_arr, T max_dim1_val, T min_dim2_val)
		{
			if (root == nullptr)
			{
				std::cout << "Tree is empty; nothing to search\n";
				res_arr = nullptr;
				return;
			}
			size_t res_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			res_arr = new RetType[res_arr_size];
			num_res_elems = 0;
			twoSidedLeftSearchRecur(res_arr, num_res_elems, res_arr_size, *root, max_dim1_val, min_dim2_val);

			// Ensure that no more memory is taken up than needed
			if (res_arr_size > num_res_elems)
				resizeArray(res_arr, res_arr_size, num_res_elems);
		};
		template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedRightSearch(size_t &num_res_elems, RetType *&res_arr, T min_dim1_val, T min_dim2_val)
		{
			if (root == nullptr)
			{
				std::cout << "Tree is empty; nothing to search\n";
				res_arr = nullptr;
				return;
			}
			size_t res_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			res_arr = new RetType[res_arr_size];
			num_res_elems = 0;
			twoSidedRightSearchRecur(res_arr, num_res_elems, res_arr_size, *root, min_dim1_val, min_dim2_val);

			// Ensure that no more memory is taken up than needed
			if (res_arr_size > num_res_elems)
				resizeArray(res_arr, res_arr_size, num_res_elems);
		};

	private:
		// Want unique copies of each tree, so no assignment or copying allowed
		StaticPSTCPURecur& operator=(StaticPSTCPURecur &tree);	// assignment operator
		StaticPSTCPURecur(StaticPSTCPURecur &tree);	// copy constructor

		// Declare nested class and define in implementation file
		class CPURecurTreeNode;

		CPURecurTreeNode *root;

		// Recursive constructor helper to populate the tree
		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		// Note that **const ensures that a second-level pointer (**) is constant, but that the pointer to which it points (i.e. a first-level pointer, *) is not
		void populateTreeRecur(CPURecurTreeNode &subtree_root, PointStructTemplate<T, IDType, num_IDs> *const *const dim1_val_ptr_subarr, PointStructTemplate<T, IDType, num_IDs> *const *const dim2_val_ptr_subarr, const size_t num_elems);
		// Returns index in dim1_val_ptr_arr of elem_to_find
		long long binarySearch(PointStructTemplate<T, IDType, num_IDs> *const *const dim1_val_ptr_arr, PointStructTemplate<T, IDType, num_IDs> &elem_to_find, const size_t num_elems);

		void printRecur(std::ostream &os, const CPURecurTreeNode &subtree_root, std::string prefix, std::string child_prefix) const;

		// Search-related helper functions
		template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void threeSidedSearchRecur(RetType *&res_arr, size_t &num_res_elems, size_t &res_arr_size, CPURecurTreeNode &subtree_root, T min_dim1_val, T max_dim1_val, T min_dim2_val);
		template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedLeftSearchRecur(RetType *&res_arr, size_t &num_res_elems, size_t &res_arr_size, CPURecurTreeNode &subtree_root, T max_dim1_val, T min_dim2_val);
		template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedRightSearchRecur(RetType *&res_arr, size_t &num_res_elems, size_t &res_arr_size, CPURecurTreeNode &subtree_root, T min_dim1_val, T min_dim2_val);
		template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void reportAllNodes(RetType *&res_arr, size_t &num_res_elems, size_t &res_arr_size, CPURecurTreeNode &subtree_root, T min_dim2_val);

	// Allow printing operator << to be declared for CPURecurTreeNode
	// For friend functions of template classes, for the compiler to recognise the function as a template function, it is necessary to either pre-declare each template friend function before the template class and modify the class-internal function declaration with an additional <> between the operator and the parameter list; or to simply define the friend function when it is declared
	//	https://isocpp.org/wiki/faq/templates#template-friends
	friend std::ostream &operator<<(std::ostream &os, const CPURecurTreeNode &tn)
	{
		tn.print(os);
		return os;
	};
};

// Implementation file; for class templates, implementations must be in the same file as the declaration so that the compiler can access them
#include "static-pst-cpu-recur.tpp"

#endif
