#ifndef STATIC_PST_GPU_H
#define STATIC_PST_GPU_H

#include <cooperative_groups.h>
#include <ostream>
#include <type_traits>

#include "class-member-checkers.h"
#include "gpu-err-chk.h"
#include "power-of-2-functions.h"
#include "static-priority-search-tree.h"
#include "type-concepts.h"


// To use __global__ function as a friend, must not define it at the same time as it is declared
// As references passed to a global function live on host code, references to variables are not valid if the value does not reside in pinned memory
// As function definition uses StaticPSTGPU members, must define function after StaticPSTGPU has been defined (hence the placement of the implementation file after the class declaration), whereas template friend declaration within class requires prior knowledge of existence (declaration) of template function
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__global__ void populateTree(T *const root_d, const size_t num_elem_slots,
								PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
								size_t *const dim1_val_ind_arr_d,
								size_t *dim2_val_ind_arr_d,
								size_t *dim2_val_ind_arr_secondary_d,
								const size_t val_ind_arr_start_ind,
								const size_t num_elems,
								const size_t target_node_start_ind);

// C++ allows trailing template type arguments and function parameters to have default values; for template type arguments, it is forbidden for default arguments to be specified for a class template member outside of the class template; for function parameters, one must not declare the default arguments again (as it is regarded as a redefinition, even if the values are the same)

// Cannot overload a global function over a host function, even if the number of arguments differs
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
__global__ void threeSidedSearchGlobal(T *const root_d, const size_t num_elem_slots,
										const long long start_node_ind,
										RetType *const res_arr_d,
										const T min_dim1_val, const T max_dim1_val,
										const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
__global__ void twoSidedLeftSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const long long start_node_ind,
											RetType *const res_arr_d,
											const T max_dim1_val, const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
__global__ void twoSidedRightSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const long long start_node_ind,
											RetType *const res_arr_d,
											const T min_dim1_val, const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
__global__ void reportAboveGlobal(T *const root_d, const size_t num_elem_slots,
										const long long start_node_ind,
										RetType *const res_arr_d,
										const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType=void, size_t num_IDs=0>
// public superclass means that all public and protected members of base-class retain their access status in the subclass
class StaticPSTGPU: public StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_IDs>
{
	public:
		// {} is value-initialisation; for structs, this is zero-initialisation
		StaticPSTGPU(PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d, size_t num_elems,
						const int warp_multiplier=1, int dev_ind=0, int num_devs=1,
						cudaDeviceProp dev_props={});
		// Since arrays were allocated continguously, only need to free one of the array pointers
		virtual ~StaticPSTGPU()
		{
			if (num_elems != 0)
				gpuErrorCheck(cudaFree(root_d),
								"Error in freeing array storing PST on device "
								+ std::to_string(dev_ind + 1) + " (1-indexed) of "
								+ std::to_string(num_devs) + ": "
							);
		};


		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const;


		int getDevInd() const {return dev_ind;};
		cudaDeviceProp getDevProps() const {return dev_props;};
		int getNumDevs() const {return num_devs;};


		// Public search functions

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void threeSidedSearch(size_t &num_res_elems, RetType *&res_arr_d,
								T min_dim1_val, T max_dim1_val, T min_dim2_val,
								const int warp_multiplier=1);

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedLeftSearch(size_t &num_res_elems, RetType *&res_arr_d,
								T max_dim1_val, T min_dim2_val,
								const int warp_multiplier=1);

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedRightSearch(size_t &num_res_elems, RetType *&res_arr_d,
									T min_dim1_val, T min_dim2_val,
									const int warp_multiplier=1);


		// Calculate minimum amount of global memory that must be available for allocation on the GPU for construction and search to run correctly
		static size_t calcGlobalMemNeeded(const size_t num_elems);


		// Functor (callable object) used instead of nested __host__ __device__ lambdas, as such lambdas are not permitted within other __host__ __device__ lambdas
		// Must be public to be accessible in __global__ functions
		struct Dim1ValIndCompIncOrd
		{
			Dim1ValIndCompIncOrd(PointStructTemplate<T, IDType, num_IDs> *pt_arr_d) : pt_arr_d(pt_arr_d) {};

			__host__ __device__ bool operator()(const size_t &i, const size_t &j)
			{
				return pt_arr_d[i].compareDim1(pt_arr_d[j]) < 0;
			};

			private:
				PointStructTemplate<T, IDType, num_IDs> *pt_arr_d;
		};
		struct Dim2ValIndCompDecOrd
		{
			Dim2ValIndCompDecOrd(PointStructTemplate<T, IDType, num_IDs> *pt_arr_d) : pt_arr_d(pt_arr_d) {};

			__host__ __device__ bool operator()(const size_t &i, const size_t &j)
			{
				return pt_arr_d[i].compareDim2(pt_arr_d[j]) > 0;
			};

			private:
				PointStructTemplate<T, IDType, num_IDs> *pt_arr_d;
		};

	private:
		// Want unique copies of each tree, so no assignment or copying allowed
		StaticPSTGPU& operator=(StaticPSTGPU &tree);	// assignment operator
		StaticPSTGPU(StaticPSTGPU &tree);	// copy constructor


		// Construction-related helper functions

		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		// Returns index in dim1_val_ind_arr_d of elem_to_find
		// Must be a static function because it is called during construction
		__forceinline__ __device__ static long long binarySearch(PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
																	size_t *const dim1_val_ind_arr_d,
																	const PointStructTemplate<T, IDType, num_IDs> &elem_to_find,
																	const size_t init_ind,
																	const size_t num_elems
																);

		__forceinline__ __device__ static void constructNode(T *const root_d,
																const size_t num_elem_slots,
																PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
																const size_t target_node_ind,
																size_t *const dim1_val_ind_arr_d,
																size_t *const dim2_val_ind_arr_d,
																size_t *const dim2_val_ind_arr_secondary_d,
																const size_t max_dim2_val_dim1_array_ind,
																const size_t subelems_start_ind,
																const size_t num_subelems,
																size_t &left_subarr_num_elems,
																size_t &right_subarr_start_ind,
																size_t &right_subarr_num_elems);

		__forceinline__ __device__ static void setNode(T *const root_d,
														const size_t node_ind,
														const size_t num_elem_slots,
														const PointStructTemplate<T, IDType, num_IDs> &source_data,
														const T median_dim1_val)
		{
			getDim1ValsRoot(root_d, num_elem_slots)[node_ind] = source_data.dim1_val;
			getDim2ValsRoot(root_d, num_elem_slots)[node_ind] = source_data.dim2_val;
			getMedDim1ValsRoot(root_d, num_elem_slots)[node_ind] = median_dim1_val;
			if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
				getIDsRoot(root_d, num_elem_slots)[node_ind] = source_data.id;
		};


		// Search-related helper functions

		// Helper functions for determining how to delegate work during searches
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void do3SidedSearchDelegation(const unsigned char curr_node_bitcode,
																		T *const root_d,
																		const size_t num_elem_slots,
																		RetType *const res_arr_d,
																		const T min_dim1_val,
																		const T max_dim1_val,
																		const T curr_node_med_dim1_val,
																		const T min_dim2_val,
																		long long &search_ind,
																		long long *const search_inds_arr,
																		unsigned char &search_code,
																		unsigned char *const search_codes_arr
																	);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void doLeftSearchDelegation(const bool range_split_poss,
																		const unsigned char curr_node_bitcode,
																		T *const root_d,
																		const size_t num_elem_slots,
																		RetType *const res_arr_d,
																		const T min_dim2_val,
																		long long &search_ind,
																		long long *const search_inds_arr,
																		unsigned char &search_code,
																		unsigned char *const search_codes_arr
																	);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void doRightSearchDelegation(const bool range_split_poss,
																		const unsigned char curr_node_bitcode,
																		T *const root_d,
																		const size_t num_elem_slots,
																		RetType *const res_arr_d,
																		const T min_dim2_val,
																		long long &search_ind,
																		long long *const search_inds_arr,
																		unsigned char &search_code,
																		unsigned char *const search_codes_arr
																	);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void doReportAboveDelegation(const unsigned char curr_node_bitcode,
																T *const root_d,
																const size_t num_elem_slots,
																RetType *const res_arr_d,
																const T min_dim2_val,
																long long &search_ind,
																long long *const search_inds_arr,
																unsigned char *const search_codes_arr = nullptr);

		// Helper functions for delegating work during searches
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void splitLeftSearchWork(T *const root_d,
																	const size_t num_elem_slots,
																	const size_t target_node_ind,
																	RetType *const res_arr_d,
																	const T max_dim1_val,
																	const T min_dim2_val,
																	long long *const search_inds_arr,
																	unsigned char *const search_codes_arr
																);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void splitReportAboveWork(T *const root_d,
																		const size_t num_elem_slots,
																		const size_t target_node_ind,
																		RetType *const res_arr_d,
																		const T min_dim2_val,
																		long long *const search_inds_arr,
																		unsigned char *const search_codes_arr = nullptr
																	);

		// Helper function for threads to determine whether all iterations have ended
		__forceinline__ __device__ static void detNextIterState(long long &search_ind,
																long long *const search_inds_arr,
																bool &cont_iter,
																unsigned char *const search_code = nullptr,
																unsigned char *const search_codes_arr = nullptr
															);


		// Data-accessing helper functions

		// Helper functions for getting start indices for various arrays
		__forceinline__ __host__ __device__ static T* getDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root;};
		__forceinline__ __host__ __device__ static T* getDim2ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + num_elem_slots;};
		__forceinline__ __host__ __device__ static T* getMedDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + (num_elem_slots << 1);};
		__forceinline__ __host__ __device__ static IDType* getIDsRoot(T *const root, const size_t num_elem_slots)
			requires NonVoidType<IDType>
		{
			const size_t val_subarr_offset_bytes = num_elem_slots * sizeof(T) * num_val_subarrs;
			const size_t val_subarr_offset_IDTypes = val_subarr_offset_bytes / sizeof(IDType)
														+ (val_subarr_offset_bytes % sizeof(IDType) == 0 ? 0 : 1);
			return reinterpret_cast<IDType *>(root) + val_subarr_offset_IDTypes;
		};
		__forceinline__ __host__ __device__ static unsigned char* getBitcodesRoot(T *const root, const size_t num_elem_slots)
		// Use reinterpret_cast for pointer conversions
		{
			if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
				// Argument of cast is of type T *
				return reinterpret_cast<unsigned char*>(root + num_val_subarrs * num_elem_slots);
			else
				// Argument of cast is of type IDType *
				return reinterpret_cast<unsigned char*>(getIDsRoot(root, num_elem_slots) + num_IDs * num_elem_slots);
		};


		// Data footprint calculation functions

		// Helper function for calculating minimum number of array slots necessary to construct a complete tree with num_elems elements
		inline static size_t calcNumElemSlots(const size_t num_elems)
		{
			/*
				Minimum number of array slots necessary to construct any complete tree with num_elems elements is 1 less than the smallest power of 2 greater than num_elems
				Tree is fully balanced by construction, with the placement of nodes in the partially empty last row being deterministic, but input-dependent
			*/
			// Number of element slots in each container subarray is minPowerOf2GreaterThan(num_elems) - 1
			return minPowerOf2GreaterThan(num_elems) - 1;
		};

		// Calculate size of array allocated for the tree in units of number of elements of type T or IDType, whichever is larger
		inline static size_t calcTotArrSizeNumMaxDataIDTypes(const size_t num_elems);	

		// Helper function for calculating the number of elements of size T necessary to instantiate an array for root of trees with no ID field
		template <size_t num_T_subarrs>
		inline static size_t calcTotArrSizeNumTs(const size_t num_elem_slots);

		// Helper function for calculating the number of elements of size IDType necessary to instantiate an array for root of tree; calculation differs from calcTotArrSizeNumTs() due to need for IDType alignment to be satisfied when sizeof(IDType) > sizeof(T)
		template <size_t num_T_subarrs>
			requires NonVoidType<IDType>
		inline static size_t calcTotArrSizeNumIDTypes(const size_t num_elem_slots);


		void printRecur(std::ostream &os, T *const tree_root, const size_t curr_ind,
							const size_t num_elem_slots, std::string prefix,
							std::string child_prefix
						) const;


		// Data members

		/*
			Implicit tree structure, with field-major orientation of nodes; all values are stored in one contiguous array on device
			Implicit subdivisions:
				T *dim1_vals_root_d;
				T *dim2_vals_root_d;
				T *med_dim1_vals_root_d;
				(If PointStructs have an .id field:
					IDType *ids_root_d;)
				unsigned char *bitcodes_root_d;
		*/
		T *root_d;
		size_t num_elem_slots;	// Allocated size of each subarray of values
		// Number of actual elements in tree; maintained because num_elem_slots - num_elems could be up to 2 * num_elems if there is only one element in the final level
		size_t num_elems;

		// Save GPU info for later usage
		int dev_ind;
		cudaDeviceProp dev_props;
		int num_devs;

		// Number of working arrays necessary to construct the tree: 1 array of dim1_val indices, 2 arrays for dim2_val indices (one that is the input, one that is the output after dividing up the indices between the current node's two children; this switches at every level of the tree)
		const static unsigned char num_constr_working_arrs = 3;
		// 1 subarray each for dim1_val, dim2_val and med_dim1_val
		const static unsigned char num_val_subarrs = 3;

		// Without explicit instantiation, enums do not take up any space
		// To save shared memory usage, reportAboveGlobal() has no search_codes array, so use the search index array to indicate an inactive thread instead
		enum IndexCodes
		{
			INACTIVE_IND = -1
		};

		enum SearchCodes
		{
			REPORT_ABOVE,
			LEFT_SEARCH,
			RIGHT_SEARCH,
			THREE_SEARCH
		};


	// Friend function declarations

	/*
		For friend functions of template classes, for the compiler to recognise the function as a template function, it is necessary to either pre-declare each template friend function before the template class; or to simply define the friend function when it is declared
		https://isocpp.org/wiki/faq/templates#template-friends

		Note that <> means a (full) specialisation with default arguments, which in this case are the template parameters of the enclosing class
	*/
	friend __global__ void populateTree <> (T *const root_d, const size_t num_elem_slots,
											PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
											size_t *const dim1_val_ind_arr_d, size_t *dim2_val_ind_arr_d,
											size_t *dim2_val_ind_arr_secondary_d,
											const size_t val_ind_arr_start_ind, const size_t num_elems,
											const size_t target_node_start_ind
										);

	/*
		As partial specialisation is not allowed (i.e. mixing the (already-instantiated) template types of the enclosing class and the still-generic template type RetType); either replace already-declared types with generic type placeholders (to not overshadow the enclosing template types) and without requires clauses (due to the constraint imposed by C++ specification 13.7.5, point 9); or opt for full specialisation, i.e. replacing RetType with the explicit desired return types.

		Note that as generic type placeholders are used here, no <> specialisation notation is used.

		Cited: C++ specification 13.7.5 (Templates > Template declarations > Friends):
	   		Example in point 1.4: allowing for template friend functions to template classes
			Point 9: template friend declaration with a constraint depending on a template parameter from an enclosing template shall be a definition that does not declare the same function template as any function template in any other scope
	*/
	template <typename U, template<typename, typename, size_t> class PtStructTempl, typename IDT, size_t NIDs, typename RetType>
	friend __global__ void threeSidedSearchGlobal(U *const root_d,
													const size_t num_elem_slots,
													const long long start_node_ind,
													RetType *const res_arr_d,
													const U min_dim1_val,
													const U max_dim1_val,
													const U min_dim2_val
												);

	template <typename U, template<typename, typename, size_t> class PtStructTempl, typename IDT, size_t NIDs, typename RetType>
	friend __global__ void twoSidedLeftSearchGlobal(U *const root_d,
													const size_t num_elem_slots,
													const long long start_node_ind,
													RetType *const res_arr_d,
													const U max_dim1_val,
													const U min_dim2_val
												);

	template <typename U, template<typename, typename, size_t> class PtStructTempl, typename IDT, size_t NIDs, typename RetType>
	friend __global__ void twoSidedRightSearchGlobal(U *const root_d,
														const size_t num_elem_slots,
														const long long start_node_ind,
														RetType *const res_arr_d,
														const U min_dim1_val,
														const U min_dim2_val
													);

	template <typename U, template<typename, typename, size_t> class PtStructTempl, typename IDT, size_t NIDs, typename RetType>
	friend __global__ void reportAboveGlobal(U *const root_d,
												const size_t num_elem_slots,
												const long long start_node_ind,
												RetType *const res_arr_d,
												const U min_dim2_val
											);
};

// Implementation file; for class templates, implementations must be in the same file as the declaration so that the compiler can access them
#include "static-pst-gpu-populate-tree.tu"
#include "static-pst-gpu-search-functions.tu"
#include "static-pst-gpu.tu"

#endif
