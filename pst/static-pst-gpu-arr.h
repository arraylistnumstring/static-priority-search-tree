#ifndef STATIC_PST_GPU_ARR_H
#define STATIC_PST_GPU_ARR_H

#include "dev-symbols.h"			// For global memory-scoped variable res_arr_ind_d
#include "gpu-err-chk.h"
#include "gpu-power-of-2-functions.h"
#include "static-priority-search-tree.h"
#include "type-concepts.h"


// To use __global__ function as a friend, must not define it at the same time as it is declared
// As references passed to a global function live on host code, references to variables are not valid if the value does not reside in pinned memory
// As function definition uses StaticPSTGPUArr members, must define function after StaticPSTGPUArr has been defined (hence the placement of the implementation file after the class declaration
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__global__ void populateTrees(T *const tree_arr_d, const size_t full_tree_num_elem_slots,
								PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
								size_t *const dim1_val_ind_arr_d,
								size_t *dim2_val_ind_arr_d,
								size_t *dim2_val_ind_arr_secondary_d,
								const size_t num_elems);

// Array of shallow on-GPU PSTs that do not require dynamic parallelism to construct or search
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
		 	typename IDType=void, size_t num_IDs=0>
class StaticPSTGPUArr: public StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_IDs>
{
	public:
		// {} is value-initialisation; for structs, this is zero-initialisation
		StaticPSTGPUArr(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d, size_t num_elems,
							const unsigned threads_per_block, int dev_ind=0, int num_devs=1,
							cudaDeviceProp dev_props={}
						);
		// Since arrays were allocated continguously, only need to free one of the array pointers
		virtual ~StaticPSTGPUArr()
		{
			if (num_elems != 0)
				gpuErrorCheck(cudaFree(tree_arr_d),
								"Error in freeing array of PSTs on device "
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

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
					requires std::disjunction<
										std::is_same<RetType, IDType>,
										std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
						>::value
		void threeSidedSearch(size_t &num_res_elems, RetType *&res_arr_d,
								T min_dim1_val, T max_dim1_val, T min_dim2_val);

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
					requires std::disjunction<
										std::is_same<RetType, IDType>,
										std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
						>::value
		void twoSidedLeftSearch(size_t &num_res_elems, RetType *&res_arr_d,
								T max_dim1_val, T min_dim2_val);

		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
					requires std::disjunction<
										std::is_same<RetType, IDType>,
										std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
						>::value
		void twoSidedRightSearch(size_t &num_res_elems, RetType *&res_arr_d,
									T min_dim1_val, T min_dim2_val);


		// Calculate minimum amount of global memory that must be available for allocation on the GPU for construction and search to run correctly
		static size_t calcGlobalMemNeeded(const size_t num_elems, const unsigned threads_per_block);

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
		// Want unique copies of each tree array, so no assignment or copying allowed
		StaticPSTGPUArr& operator=(StaticPSTGPUArr &tree_arr);	//assignment operator
		StaticPSTGPUArr(StaticPSTGPUArr &tree_arr);		// copy constructor

		
		// Construction-related helper functions

		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		// Returns index in dim1_val_ind_arr of elem_to_find
		__forceinline__ __device__ static long long binarySearch(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d,
																	size_t *const &dim1_val_ind_arr_d,
																	PointStructTemplate<T, IDType, num_IDs> &elem_to_find,
																	const size_t &init_ind,
																	const size_t &num_elems
																);


		// Data footprint calculation functions

		// Helper function for calculating minimum number of array slots necessary to construct a complete tree with num_elems elements
		// Must be a static function because it is called during construction
		static size_t calcNumElemSlotsPerTree(const size_t num_elems_per_tree)
		{
			// Minimum number of array slots necessary to construct any complete tree with num_elems elements is 1 less than the smallest power of 2 greater than num_elems
			// Number of elements in each container subarray for each tree is nextGreaterPowerOf2(num_elems) - 1
			return nextGreaterPowerOf2(num_elems_per_tree) - 1;
		};

		// Calculate size of array of trees in units of number of elements of type T or IDType, whichever is larger
		static size_t calcTreeArrSizeNumMaxDataIDTypes(const size_t num_elems, const unsigned threads_per_block);

		// Calculate size of array allocated for each tree in units of number of elements of type T or IDType, whichever is larger
		static size_t calcTreeSizeNumMaxDataIDTypes(const size_t num_elem_slots_per_tree);

		// Helper function for calculating the number of elements of size T necessary to instantiate each tree for trees with no ID field
		template <size_t num_T_subarrs>
		static size_t calcTreeSizeNumTs(const size_t num_elem_slots_per_tree);

		// Helper function for calculating the number of elements of size U necessary to instantiate each tree, for data types U and V such that sizeof(U) >= sizeof(V)
		template <typename U, size_t num_U_subarrs, typename V, size_t num_V_subarrs>
			requires SizeOfUAtLeastSizeOfV<U, V>
		static size_t calcTreeSizeNumUs(const size_t num_elem_slots_per_tree);


		// Data members

		/*
			Implicit tree structure for each tree, with field-major orientation of nodes; all values are stored in one contiguous array on device
			Implicit subdivisions:
				T *dim1_vals_root_d;
				T *dim2_vals_root_d;
				T *med_dim1_vals_root_d;
				(Optional:
					IDType *ids_root_d;)
				unsigned char *bitcodes_root_d;
		*/
		T *tree_arr_d;
		size_t full_tree_num_elem_slots;
		// Necessary to allow each thread block to calculate the correct starting location of its assigned tree
		// Must be calculated in units of the maximally-sized datatype in order to ensure that each tree obeys its alignment requirements for both its data and ID types
		size_t full_tree_size_num_max_data_id_types;
		size_t num_elems;

		// unsigned type is chosen to correspond with CUDA on-device data type of fields of gridDim and blockDim, respectively
		// Number of thread blocks in each kernel call (and therefore the number of shallow trees)
		unsigned num_thread_blocks;
		// Number of threads per block (and therefore per shallow tree, as each such tree is processed by one thread block)
		unsigned threads_per_block;

		//Save GPU info for later usage
		int dev_ind;
		cudaDeviceProp dev_props;
		int num_devs;

		// Number of working arrays necessary per tree: 1 array of dim1_val indices, 2 arrays for dim2_val indices (one that is the input, one that is the output after dividing up the indices between the current node's two children; this switches at every level of the tree)
		const static unsigned char num_constr_working_arrs = 3;
		// 1 subarray each for dim1_val, dim2_val and med_dim1_val
		const static unsigned char num_val_subarrs = 3;

		// Without explicit instantiation, enums do not take up any space
		enum SearchCodes
		{
			REPORT_ALL,
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
	friend __global__ void populateTrees <> (T *const tree_arr_d, const size_t full_tree_num_elem_slots,
												PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
												size_t *const dim1_val_ind_arr_d,
												size_t *dim2_val_ind_arr_d,
												size_t *dim2_val_ind_arr_secondary_d,
												const size_t num_elems
											);

};

#include "static-pst-gpu-arr.tu"
#include "static-pst-gpu-arr-populate-trees.tu"

#endif
