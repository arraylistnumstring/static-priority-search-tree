#ifndef STATIC_PST_CPU_ITER_H
#define STATIC_PST_CPU_ITER_H

#include <stack>

#include "class-member-checkers.h"
#include "cpu-iter-tree-node.h"
#include "static-priority-search-tree.h"
#include "type-concepts.h"


template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void populateTree(T *const root, const size_t num_elem_slots,
					PointStructTemplate<T, IDType, num_IDs> *const pt_arr,
					size_t *const dim1_val_ind_arr, size_t *dim2_val_ind_arr,
					size_t *dim2_val_ind_arr_secondary,
					const size_t start_ind, const size_t num_elems);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType=void, size_t num_IDs=0>
// public superclass means that all public and protected members of base-class retain their access status in the subclass
class StaticPSTCPUIter : public StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_IDs>
{
	public:
		StaticPSTCPUIter(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr, size_t num_elems);
		// Since arrays were allocated continguously, only need to free one of the array pointers
		virtual ~StaticPSTCPUIter() {if (num_elems != 0) delete[] root;};


		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const;


		// Public search functions

		// Uses stacks instead of recursion or dynamic parallelism
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void threeSidedSearch(size_t &num_res_elems, RetType *&res_arr, T min_dim1_val,
								T max_dim1_val, T min_dim2_val);

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedLeftSearch(size_t &num_res_elems, RetType *&res_arr, T max_dim1_val, T min_dim2_val);

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedRightSearch(size_t &num_res_elems, RetType *&res_arr, T min_dim1_val, T min_dim2_val);

	private:
		// Want unique copies of each tree, so no assignment or copying allowed
		StaticPSTCPUIter& operator=(StaticPSTCPUIter &tree);	// assignment operator
		StaticPSTCPUIter(StaticPSTCPUIter &tree);	// copy constructor


		// Construction-related helper functions

		static void constructNode(T *const &root,
									const size_t &num_elem_slots,
									PointStructTemplate<T, IDType, num_IDs> *const &pt_arr,
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
									size_t &right_subarr_num_elems
								);

		static void setNode(T *const root, const size_t node_ind, const size_t num_elem_slots,
							PointStructTemplate<T, IDType, num_IDs> &source_data, T median_dim1_val)
		{
			getDim1ValsRoot(root, num_elem_slots)[node_ind] = source_data.dim1_val;
			getDim2ValsRoot(root, num_elem_slots)[node_ind] = source_data.dim2_val;
			getMedDim1ValsRoot(root, num_elem_slots)[node_ind] = median_dim1_val;
			if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
				getIDsRoot(root, num_elem_slots)[node_ind] = source_data.id;
		};


		// Search-related helper functions

		// Helper functions for tracking work to be completed
		void do3SidedSearchDelegation(const unsigned char &curr_node_bitcode, T min_dim1_val,
										T max_dim1_val, T curr_node_med_dim1_val,
										const long long &search_ind,
										std::stack<long long> &search_inds_stack,
										std::stack<unsigned char> &search_codes_stack
									);
		void doLeftSearchDelegation(const bool range_split_poss,
									const unsigned char &curr_node_bitcode,
									const long long &search_ind,
									std::stack<long long> &search_inds_stack,
									std::stack<unsigned char> &search_codes_stack
								);
		void doRightSearchDelegation(const bool range_split_poss,
										const unsigned char &curr_node_bitcode,
										const long long &search_ind,
										std::stack<long long> &search_inds_stack,
										std::stack<unsigned char> &search_codes_stack
									);
		void doReportAllNodesDelegation(const unsigned char &curr_node_bitcode,
										const long long &search_ind,
										std::stack<long long> &search_inds_stack,
										std::stack<unsigned char> &search_codes_stack
									);


		// Data-accessing helper functions

		// Helper functions for getting start indices for various arrays
		static T* getDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root;};
		static T* getDim2ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + num_elem_slots;};
		static T* getMedDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + (num_elem_slots << 1);};
		static IDType* getIDsRoot(T *const root, const size_t num_elem_slots)
			{return reinterpret_cast<IDType *>(root + num_elem_slots * num_val_subarrs);};
		static unsigned char* getBitcodesRoot(T *const root, const size_t num_elem_slots)
			// Use reinterpret_cast for pointer conversions
			{
				if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
					// Argument of cast is of type T *
					return reinterpret_cast<unsigned char*>(root + num_val_subarrs * num_elem_slots);
				else
					// Argument of cast is of type IDType *
					return reinterpret_cast<unsigned char*>(getIDsRoot(root, num_elem_slots) + num_ID_subarrs * num_elem_slots);
			};	


		// Data footprint calculation functions

		// Helper function for calculating the number of elements of size T necessary to instantiate an array for root of tree
		template <size_t num_T_subarrs>
		inline static size_t calcTotArrSizeNumTs(const size_t num_elem_slots);

		// Helper function for calculating the number of elements of size IDType necessary to instantiate an array for root of tree; calculation differs from calcTotArrSizeNumTs() due to need for IDType alignment to be satisfied when sizeof(IDType) > sizeof(T)
		template <size_t num_T_subarrs>
			requires NonVoidType<IDType>
		inline static size_t calcTotArrSizeNumIDTypes(const size_t num_elem_slots);

		// Helper function for calculating the next power of 2 greater than num
		template <typename U>
			requires std::unsigned_integral<U>
		static U nextGreaterPowerOf2(const U num)
			{return 1 << expOfNextGreaterPowerOf2(num);};
		template <typename U>
			requires std::unsigned_integral<U>
		static U expOfNextGreaterPowerOf2(const U num);

		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		// Returns index in dim1_val_ind_arr of elem_to_find
		static long long binarySearch(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr,
										size_t *const &dim1_val_ind_arr,
										PointStructTemplate<T, IDType, num_IDs> &elem_to_find,
										const size_t &init_ind, const size_t &num_elems
									);


		void printRecur(std::ostream &os, T *const &tree_root, const size_t curr_ind,
						const size_t num_elem_slots, std::string prefix, std::string child_prefix
					) const;


		// Data members

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

		// 1 subarray each for dim1_val, dim2_val and med_dim1_val
		const static size_t num_val_subarrs = 3;
		const static size_t num_ID_subarrs = num_IDs;

		// Without an explicit instantiation, enums don't take up any space
		// enum name : type gives the enum an explicit underlying type; type chosen to correspond to that given to the enum SearchCodes of StaticPSTGPU
		enum SearchCodes : unsigned char
		{
			REPORT_ALL,
			LEFT_SEARCH,
			RIGHT_SEARCH,
			THREE_SEARCH
		};


	// Friend function declarations

	/*
		For friend functions of template classes, for the compiler to recognise the function as a template function, it is necessary to either pre-declare each template friend function before the template class and modify the class-internal function declaration with an additional <> between the operator and the parameter list; or to simply define the friend function when it is declared
		https://isocpp.org/wiki/faq/templates#template-friends
	*/
	friend void populateTree <> (T *const root, const size_t num_elem_slots,
									PointStructTemplate<T, IDType, num_IDs> *const pt_arr,
									size_t *const dim1_val_ind_arr,
									size_t *dim2_val_ind_arr, size_t *dim2_val_ind_arr_secondary,
									const size_t start_ind, const size_t num_elems);
};

// Implementation file; for class templates, implementations must be in the same file as the declaration so that the compiler can access them
#include "static-pst-cpu-iter-populate-tree.tpp"
#include "static-pst-cpu-iter.tpp"

#endif
