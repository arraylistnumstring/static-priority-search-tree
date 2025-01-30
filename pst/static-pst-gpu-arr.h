#ifndef STATIC_PST_GPU_ARR_H
#define STATIC_PST_GPU_ARR_H

#include "dev-symbols.h"		// For global memory-scoped variable res_arr_ind_d
#include "gpu-err-chk.h"
#include "gpu-tree-node.h"

// Array of shallow on-GPU PSTs that do not require dynamic parallelism to construct or search
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
		 	typename IDType=void, size_t num_IDs=0>
class StaticPSTGPUArr: public StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_IDs>
{
	public:
		// {} is value-initialisation; for structs, this is zero-initialisation
		StaticPSTGPUArr(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d, size_t num_elems,
							const int warps_per_block=1, int dev_ind=0, int num_devs=1,
							cudaDeviceProp dev_props={}
						);
		// Since arrays were allocated continguously, only need to free one of the array pointers
		virtual ~StaticPSTGPUArr()
		{
			/*
			if (num_elems != 0)	// TODO: modify condition here as is appropriate
				gpuErrorCheck(cudaFree(root_d),
								"Error in freeing array storing on-device PST array on device "
								+ std::to_string(dev_ind + 1) + " (1-indexed) of "
								+ std::to_string(num_devs) + ": "
							);
			*/
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
										std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>
						>::value
		void threeSidedSearch(size_t &num_res_elems, RetType *&res_arr_d,
								T min_dim1_val, T max_dim1_val, T min_dim2_val)
		{
		};
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
					requires std::disjunction<
										std::is_same<RetType, IDType>,
										std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>
						>::value
		void twoSidedLeftSearch(size_t &num_res_elems, RetType *&res_arr_d,
								T max_dim1_val, T min_dim2_val)
		{
		};
		template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
					requires std::disjunction<
										std::is_same<RetType, IDType>,
										std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>
						>::value
		void twoSidedRightSearch(size_t &num_res_elems, RetType *&res_arr_d,
									T min_dim1_val, T min_dim2_val)
		{
		};

	private:
		size_t num_elems;

		//Save GPU info for later usage
		int dev_ind;
		cudaDeviceProp dev_props;
		int num_devs;

		// Number of warps per block (and therefore per shallow tree, as each such tree is processed by one thread block)
		unsigned warps_per_block;

		// Number of working arrays necessary per tree: 1 array of dim1_val indices, 2 arrays for dim2_val indices (one that is the input, one that is the output after dividing up the indices between the current node's two children; this switches at every level of the tree)
		const static unsigned char num_constr_working_arrs = 3;
		// 1 subarray each for dim1_val, dim2_val and med_dim1_val
		const static unsigned char num_val_subarrs = 3;

		enum SearchCodes
		{
			REPORT_ALL,
			LEFT_SEARCH,
			RIGHT_SEARCH,
			THREE_SEARCH
		};
};

#include "static-pst-gpu-arr.tu"

#endif
