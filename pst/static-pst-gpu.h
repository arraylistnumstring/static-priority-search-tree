#ifndef STATIC_PST_GPU_H
#define STATIC_PST_GPU_H

#include "dev-symbols.h"	// For global memory-scoped variable res_arr_ind_d
#include "gpu-err-chk.h"
#include "static-priority-search-tree.h"
#include "static-pst-concepts.h"


// To use __global__ function as a friend, must not define it at the same time as it is declared
// As references passed to a global function live on host code, references to variables are not valid if the value does not reside in pinned memory
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

// Assigning elements of an array on device such that array[i] = i
__global__ void indexAssignment(size_t *const ind_arr, const size_t num_elems);

// Cannot overload a global function over a host function, even if the number of arguments differs
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__global__ void threeSidedSearchGlobal(T *const root_d, const size_t num_elem_slots,
										const size_t start_node_ind,
										RetType *const res_arr_d,
										const T min_dim1_val, const T max_dim1_val,
										const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__global__ void twoSidedLeftSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const size_t start_node_ind,
											RetType *const res_arr_d,
											const T max_dim1_val, const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__global__ void twoSidedRightSearchGlobal(T *const root_d, const size_t num_elem_slots,
											const size_t start_node_ind,
											RetType *const res_arr_d,
											const T min_dim1_val, const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs,
			typename RetType=PointStructTemplate<T, IDType, num_IDs>
		 >
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__global__ void reportAllNodesGlobal(T *const root_d, const size_t num_elem_slots,
										const size_t start_node_ind,
										RetType *const res_arr_d,
										const T min_dim2_val);

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType=void, size_t num_IDs=0>
// public superclass means that all public and protected members of base-class retain their access status in the subclass
class StaticPSTGPU: public StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_IDs>
{
	public:
		StaticPSTGPU(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr, size_t num_elems);
		// Since arrays were allocated continguously, only need to free one of the array pointers
		virtual ~StaticPSTGPU()
		{
			if (num_elem_slots != 0)
				gpuErrorCheck(cudaFree(root_d),
								"Error in freeing array storing on-device PST on device "
								+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
								+ ": ");
		};

		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const;

		int getDevInd() const {return dev_ind;};
		cudaDeviceProp getDevProps() const {return dev_props;};
		int getNumDevs() const {return num_devs;};

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void threeSidedSearch(size_t &num_res_elems, RetType *&res_arr_d, T min_dim1_val, T max_dim1_val, T min_dim2_val)
		{
			if (num_elems == 0)
			{
				std::cout << "Tree is empty; nothing to search\n";
				num_res_elems = 0;
				res_arr_d = nullptr;
				return;
			}

			gpuErrorCheck(cudaMalloc(&res_arr_d, num_elems * sizeof(RetType)),
							"Error in allocating array to store PointStruct search result on device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");
			// Set on-device global result array index to 0
			unsigned long long res_arr_ind = 0;
			// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
			gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind, sizeof(size_t),
												0, cudaMemcpyDefault),
							"Error in initialising global result array index to 0 on device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");

			// Call global function for on-device search
			threeSidedSearchGlobal<<<1, warp_multiplier * dev_props.warpSize,
										warp_multiplier * dev_props.warpSize
											* (sizeof(long long) + sizeof(unsigned char))>>>
				(root_d, num_elem_slots, 0, res_arr_d, min_dim1_val, max_dim1_val, min_dim2_val);

			// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
			// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
			gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d,
												sizeof(unsigned long long), 0,
												cudaMemcpyDefault),
							"Error in copying global result array final index from device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");
		};
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedLeftSearch(size_t &num_res_elems, RetType *&res_arr_d, T max_dim1_val, T min_dim2_val)
		{
			if (num_elems == 0)
			{
				std::cout << "Tree is empty; nothing to search\n";
				num_res_elems = 0;
				res_arr_d = nullptr;
				return;
			}

			gpuErrorCheck(cudaMalloc(&res_arr_d, num_elems * sizeof(RetType)),
							"Error in allocating array to store PointStruct search result on device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");
			// Set on-device global result array index to 0
			unsigned long long res_arr_ind = 0;
			// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
			gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind, sizeof(size_t),
												0, cudaMemcpyDefault),
							"Error in initialising global result array index to 0 on device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");

			// Call global function for on-device search
			twoSidedLeftSearchGlobal<<<1, warp_multiplier * dev_props.warpSize,
										warp_multiplier * dev_props.warpSize
											* (sizeof(long long) + sizeof(unsigned char))>>>
				(root_d, num_elem_slots, 0, res_arr_d, max_dim1_val, min_dim2_val);

			// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
			// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
			gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d,
												sizeof(unsigned long long), 0,
												cudaMemcpyDefault),
							"Error in copying global result array final index from device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");
		};
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		void twoSidedRightSearch(size_t &num_res_elems, RetType *&res_arr_d, T min_dim1_val, T min_dim2_val)
		{
			if (num_elems == 0)
			{
				std::cout << "Tree is empty; nothing to search\n";
				num_res_elems = 0;
				res_arr_d = nullptr;
				return;
			}

			gpuErrorCheck(cudaMalloc(&res_arr_d, num_elems * sizeof(RetType)),
							"Error in allocating array to store PointStruct search result on device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");
			// Set on-device global result array index to 0
			unsigned long long res_arr_ind = 0;
			// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
			gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind, sizeof(size_t),
												0, cudaMemcpyDefault),
							"Error in initialising global result array index to 0 on device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");

			// Call global function for on-device search
			twoSidedRightSearchGlobal<<<1, warp_multiplier * dev_props.warpSize,
										warp_multiplier * dev_props.warpSize
											* (sizeof(long long) + sizeof(unsigned char))>>>
				(root_d, num_elem_slots, 0, res_arr_d, min_dim1_val, min_dim2_val);

			// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
			// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
			gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d,
												sizeof(unsigned long long), 0,
												cudaMemcpyDefault),
							"Error in copying global result array final index from device "
							+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
							+ ": ");
		};

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


		__forceinline__ __host__ __device__ static void setNode(T *const root_d, const size_t node_ind, const size_t num_elem_slots, PointStructTemplate<T, IDType, num_IDs> &source_data, T median_dim1_val)
		{
			getDim1ValsRoot(root_d, num_elem_slots)[node_ind] = source_data.dim1_val;
			getDim2ValsRoot(root_d, num_elem_slots)[node_ind] = source_data.dim2_val;
			getMedDim1ValsRoot(root_d, num_elem_slots)[node_ind] = median_dim1_val;
			if constexpr (num_IDs == 1)
				getIDsRoot(root_d, num_elem_slots)[node_ind] = source_data.id;
		};

		__forceinline__ __device__ static void constructNode(T *const &root_d,
																const size_t &num_elem_slots,
																PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d,
																size_t &target_node_ind,
																const size_t &num_elems,
																size_t *const &dim1_val_ind_arr_d,
																size_t *&dim2_val_ind_arr_d,
																size_t *&dim2_val_ind_arr_secondary_d,
																const size_t &max_dim2_val_dim1_array_ind,
																size_t *&subelems_start_inds_arr,
																size_t *&num_subelems_arr,
																size_t &left_subarr_num_elems,
																size_t &right_subarr_start_ind,
																size_t &right_subarr_num_elems);

		// Helper functions for determining how to delegate work during searches
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void do3SidedSearchDelegation(const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																RetType *const res_arr_d,
																const T &min_dim1_val,
																const T &max_dim1_val,
																const T &curr_node_med_dim1_val,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char &search_code,
																unsigned char *const &search_codes_arr);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void doLeftSearchDelegation(const bool range_split_poss,
																const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																RetType *const res_arr_d,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char &search_code,
																unsigned char *const &search_codes_arr);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void doRightSearchDelegation(const bool range_split_poss,
																const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																RetType *const res_arr_d,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char &search_code,
																unsigned char *const &search_codes_arr);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void doReportAllNodesDelegation(const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																RetType *const res_arr_d,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char *const &search_codes_arr = nullptr);

		// Helper functions for delegating work during searches
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void splitLeftSearchWork(T *const &root_d,
																	const size_t &num_elem_slots,
																	const size_t &target_node_ind,
																	RetType *const res_arr_d,
																	const T &max_dim1_val,
																	const T &min_dim2_val,
																	long long *const &search_inds_arr,
																	unsigned char *const &search_codes_arr);
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
		__forceinline__ __device__ static void splitReportAllNodesWork(T *const &root_d,
																		const size_t &num_elem_slots,
																		const size_t &target_node_ind,
																		RetType *const res_arr_d,
																		const T &min_dim2_val,
																		long long *const &search_inds_arr,
																		unsigned char *const &search_codes_arr = nullptr);

		// Helper function for threads to determine whether all iterations have ended
		__forceinline__ __device__ static void detInactivity(long long &search_ind,
																long long *const &search_inds_arr,
																bool &cont_iter,
																unsigned char *const search_code = nullptr,
																unsigned char *const &search_codes_arr = nullptr);

		// Helper functions for getting start indices for various arrays
		__forceinline__ __host__ __device__ static T* getDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root;};
		__forceinline__ __host__ __device__ static T* getDim2ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + num_elem_slots;};
		__forceinline__ __host__ __device__ static T* getMedDim1ValsRoot(T *const root, const size_t num_elem_slots)
			{return root + (num_elem_slots << 1);};
		__forceinline__ __host__ __device__ static IDType* getIDsRoot(T *const root, const size_t num_elem_slots)
			{return reinterpret_cast<IDType *>(root + num_elem_slots * num_val_subarrs);};
		__forceinline__ __host__ __device__ static unsigned char* getBitcodesRoot(T *const root, const size_t num_elem_slots)
			// Use reinterpret_cast for pointer conversions
			{
				if constexpr (num_IDs == 0)
					// Argument of cast is of type T *
					return reinterpret_cast<unsigned char*>(root + num_val_subarrs * num_elem_slots);
				else
					// Argument of cast is of type IDType *
					return reinterpret_cast<unsigned char*>(getIDsRoot(root, num_elem_slots) + num_IDs * num_elem_slots);
			};

		// Helper function for calculating the number of elements of size T necessary to instantiate an array for root of trees with no ID field
		__forceinline__ __host__ __device__ static size_t calcTotArrSizeNumTs(const size_t num_elem_slots, const size_t num_T_subarrs);

		// Helper function for calculating the number of elements of size U necessary to instantiate an array for root, for data types U and V such that sizeof(U) >= sizeof(V)
		template <typename U, size_t num_U_subarrs, typename V, size_t num_V_subarrs>
			requires SizeOfUAtLeastSizeOfV<U, V>
		__forceinline__ __host__ __device__ static size_t calcTotArrSizeNumUs(const size_t num_elem_slots);

		// Helper function for calculating the next power of 2 greater than num
		__forceinline__ __host__ __device__ static size_t nextGreaterPowerOf2(const size_t num)
			{return 1 << expOfNextGreaterPowerOf2(num);};
		__forceinline__ __host__ __device__ static size_t expOfNextGreaterPowerOf2(const size_t num);


		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		// Returns index in dim1_val_ind_arr of elem_to_find
		__forceinline__ __host__ __device__ static long long binarySearch(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr, size_t *const &dim1_val_ind_arr, PointStructTemplate<T, IDType, num_IDs> &elem_to_find, const size_t &init_ind, const size_t &num_elems);

		void printRecur(std::ostream &os, T *const &tree_root, const size_t curr_ind, const size_t num_elem_slots, std::string prefix, std::string child_prefix) const;


		/*
			Implicit tree structure, with field-major orientation of nodes; all values are stored in one contiguous array on device
			Implicit subdivisions:
				T *dim1_vals_root_d;
				T *dim2_vals_root_d;
				T *med_dim1_vals_root_d;
				unsigned char *bitcodes_root_d;
		*/
		T *root_d;
		size_t num_elem_slots;	// Allocated size of each subarray of values
		// Number of actual elements in tree; maintained because num_elem_slots - num_elems could be up to 2 * num_elems if there is only one element in the final level
		size_t num_elems;

		// Save GPU info for later usage
		int dev_ind;
		cudaDeviceProp dev_props;
		// Number by which to multiply the number of warps in a thread block
		const static int warp_multiplier = 1;
		int num_devs;

		// Number of working arrays necessary to construct the tree: 1 array of dim1_val indices, 2 arrays for dim2_val indices (one that is the input, one that is the output after dividing up the indices between the current node's two children; this switches at every level of the tree)
		const static unsigned char num_constr_working_arrs = 3;
		// 1 subarray each for dim1_val, dim2_val and med_dim1_val
		const static unsigned char num_val_subarrs = 3;

		// Declare helper nested class for accessing specific nodes and define in implementation file; as nested class are not attached to any particular instance of the outer class by default (i.e. are like Java's static nested classes by default), only functions contained within need to be declared as static
		class TreeNode;

		// Without an explicit instantiation, enums don't take up any space
		enum IndexCodes
		{
			INACTIVE_IND = -1
		};

		enum SearchCodes
		{
			REPORT_ALL,
			LEFT_SEARCH,
			RIGHT_SEARCH,
			THREE_SEARCH
		};

	/*
		For friend functions of template classes, for the compiler to recognise the function as a template function, it is necessary to either pre-declare each template friend function before the template class and modify the class-internal function declaration with an additional <> between the operator and the parameter list; or to simply define the friend function when it is declared
		https://isocpp.org/wiki/faq/templates#template-friends

		Note that <> means a specialisation with the default arguments, which in this case are the template parameters of the enclosing class
	*/
	friend __global__ void populateTree <> (T *const root_d, const size_t num_elem_slots,
											PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
											size_t *const dim1_val_ind_arr_d, size_t *dim2_val_ind_arr_d,
											size_t *dim2_val_ind_arr_secondary_d,
											const size_t val_ind_arr_start_ind, const size_t num_elems,
											const size_t target_node_start_ind);

	// Non-template friend
	friend __global__ void indexAssignment (size_t *const ind_arr, const size_t num_elems);

	// Default template argument omitted due to requirement that only definitions of friend functions can have them
	template <typename RetType>
		requires std::disjunction<
							std::is_same<RetType, IDType>,
							std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
			>::value
	friend __global__ void threeSidedSearchGlobal <> (T *const root_d,
														const size_t num_elem_slots,
														const size_t start_node_ind,
														RetType *const res_arr_d,
														const T min_dim1_val,
														const T max_dim1_val,
														const T min_dim2_val);
	template <typename RetType>
		requires std::disjunction<
							std::is_same<RetType, IDType>,
							std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
			>::value
	friend __global__ void twoSidedLeftSearchGlobal <> (T *const root_d,
														const size_t num_elem_slots,
														const size_t start_node_ind,
														RetType *const res_arr_d,
														const T max_dim1_val,
														const T min_dim2_val);
	template <typename RetType>
		requires std::disjunction<
							std::is_same<RetType, IDType>,
							std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
			>::value
	friend __global__ void twoSidedRightSearchGlobal <> (T *const root_d,
															const size_t num_elem_slots,
															const size_t start_node_ind,
															RetType *const res_arr_d,
															const T min_dim1_val,
															const T min_dim2_val);
	template <typename RetType>
		requires std::disjunction<
							std::is_same<RetType, IDType>,
							std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
			>::value
	friend __global__ void reportAllNodesGlobal <> (T *const root_d,
													const size_t num_elem_slots,
													const size_t start_node_ind,
													RetType *const res_arr_d,
													const T min_dim2_val);
};

// Implementation file; for class templates, implementations must be in the same file as the declaration so that the compiler can access them
#include "static-pst-gpu-construction-functions.tu"
#include "static-pst-gpu-search-functions.tu"
#include "static-pst-gpu-tree-node.tu"
#include "static-pst-gpu.tu"

#endif
