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
							const int warp_multiplier=1, int dev_ind=0, int num_devs=1,
							cudaDeviceProp dev_props={}
						);
		// Since arrays were allocated continguously, only need to free one of the array pointers
		virtual ~StaticPSTGPUArr()
		{
			if (num_elems == 0)
				gpuErrorCheck(cudaFree(root_d),
								"Error in freeing array storing on-device PST array on device "
								+ std::to_string(dev_ind + 1) + " (1-indexed) of "
								+ std::to_string(num_devs) + ": "
							);
		}

		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const;

		int getDevInd() const {return dev_ind;};
		cudaDeviceProp getDevProps() const {return dev_props;};
		int getNumDevs() const {return num_devs;};
};

#endif
