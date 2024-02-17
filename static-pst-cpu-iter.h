#ifndef STATIC_PST_CPU_ITER_H
#define STATIC_PST_CPU_ITER_H

#include <stack>

#include "point-struct-cpu-iter.h"
#include "static-priority-search-tree.h"

template <typename T>
void populateTree(T *const root, const size_t num_elem_slots, PointStructCPUIter<T> *const pt_arr, size_t *const dim1_val_ind_arr, size_t *dim2_val_ind_arr, size_t *dim2_val_ind_arr_secondary, const size_t start_ind, const size_t num_elems);

template <typename T>
// public superclass means that all public and protected members of base-class retain their access status in the subclass
class StaticPSTCPUIter : public StaticPrioritySearchTree<T>
{
	// Throws a compile-time error if T is not of arithmetic (numeric) type
	// static_assert() and std::is_arithmetic are C++11 features
	// static_assert() must have two arguments to compile on CIMS
	static_assert(std::is_arithmetic<T>::value, "Input type T not of arithmetic type");

	public:
		StaticPSTCPUIter(PointStructCPUIter<T> *const &pt_arr, size_t num_elems);
		// Since arrays were allocated continguously, only need to free one of the array pointers
		virtual ~StaticPSTCPUIter() {if (num_elem_slots != 0) free(root);};

		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const;

		// Initial input value for num_res_elems is the array initialisation size
		virtual PointStructCPUIter<T>* threeSidedSearch(size_t &num_res_elems, T min_dim1_val, T max_dim1_val, T min_dim2_val)
		{
			size_t pt_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			PointStructCPUIter<T>* pt_arr = new PointStructCPUIter<T>[pt_arr_size];
			num_res_elems = 0;
			// TODO: search

			return pt_arr;
		};
		virtual PointStructCPUIter<T>* twoSidedLeftSearch(size_t &num_res_elems, T max_dim1_val, T min_dim2_val)
		{
			size_t pt_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			PointStructCPUIter<T>* pt_arr = new PointStructCPUIter<T>[pt_arr_size];
			num_res_elems = 0;
			// TODO: search

			return pt_arr;
		};
		virtual PointStructCPUIter<T>* twoSidedRightSearch(size_t &num_res_elems, T min_dim1_val, T min_dim2_val)
		{
			size_t pt_arr_size = num_res_elems == 0 ? 10 : num_res_elems;
			PointStructCPUIter<T>* pt_arr = new PointStructCPUIter<T>[pt_arr_size];
			num_res_elems = 0;
			// TODO: search

			return pt_arr;
		};

	private:
		// Want unique copies of each tree, so no assignment or copying allowed
		StaticPSTCPUIter& operator=(StaticPSTCPUIter &tree);	// assignment operator
		StaticPSTCPUIter(StaticPSTCPUIter &tree);	// copy constructor


		static void setNode(T *const root, const size_t node_ind, const size_t num_elem_slots, PointStruct<T> &source_data, T median_dim1_val)
		{
			getDim1ValsRoot(root, num_elem_slots)[node_ind] = source_data.dim1_val;
			getDim2ValsRoot(root, num_elem_slots)[node_ind] = source_data.dim2_val;
			getMedDim1ValsRoot(root, num_elem_slots)[node_ind] = median_dim1_val;
		};

		static void constructNode(T *const &root,
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
									size_t &right_subarr_num_elems);

		// Helper functions for getting start indices for various arrays
		static T* getDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root;};
		static T* getDim2ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + num_elem_slots;};
		static T* getMedDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + (num_elem_slots << 1);};
		static unsigned char* getBitcodesRoot(T *const root, const size_t num_elem_slots)
			// Use reinterpret_cast for pointer conversions
			{return reinterpret_cast<unsigned char*> (root + num_val_subarrs * num_elem_slots);};

		// Helper function for calculating the number of elements of size T necessary to instantiate an array for root
		static size_t calcTotArrSizeNumTs(const size_t num_elem_slots);

		// Helper function for calculating the next power of 2 greater than num
		static size_t nextGreaterPowerOf2(const size_t num)
			{return 1 << expOfNextGreaterPowerOf2(num);};
		static size_t expOfNextGreaterPowerOf2(const size_t num);

		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		// Returns index in dim1_val_ind_arr of elem_to_find
		static long long binarySearch(PointStructCPUIter<T> *const &pt_arr, size_t *const &dim1_val_ind_arr, PointStructCPUIter<T> &elem_to_find, const size_t &init_ind, const size_t &num_elems);

		void printRecur(std::ostream &os, T *const &tree_root, const size_t curr_ind, const size_t num_elem_slots, std::string prefix, std::string child_prefix) const;







		/*
			Implicit tree structure, with field-major orientation of nodes; all values are stored in one contiguous array on device
			Implicit subdivisions:
				T *dim1_vals_root;
				T *dim2_vals_root;
				T *med_dim1_vals_root;
				unsigned char *bitcodes_root;
		*/
		T *root;
		size_t num_elem_slots;	// Allocated size of each subarray of values
		// Number of actual elements in tree; maintained because num_elem_slots - num_elems could be up to 2 * num_elems if there is only one element in the final level
		size_t num_elems;

		const static size_t num_val_subarrs = 3;

		// Declare helper nested class for accessing specific nodes and define in implementation file; as nested class are not attached to any particular instance of the outer class by default (i.e. are like Java's static nested classes by default), only functions contained within need to be declared as static
		class TreeNode;

	/*
		For friend functions of template classes, for the compiler to recognise the function as a template function, it is necessary to either pre-declare each template friend function before the template class and modify the class-internal function declaration with an additional <> between the operator and the parameter list; or to simply define the friend function when it is declared
		https://isocpp.org/wiki/faq/templates#template-friends
	*/
	friend void populateTree <> (T *const root, const size_t num_elem_slots, PointStructCPUIter<T> *const pt_arr, size_t *const dim1_val_ind_arr, size_t *dim2_val_ind_arr, size_t *dim2_val_ind_arr_secondary, const size_t start_ind, const size_t num_elems);
};

// Implementation file; for class templates, implementations must be in the same file as the declaration so that the compiler can access them
#include "static-pst-cpu-iter-populate-tree.tpp"
#include "static-pst-cpu-iter-tree-node.tpp"
#include "static-pst-cpu-iter.tpp"

#endif
