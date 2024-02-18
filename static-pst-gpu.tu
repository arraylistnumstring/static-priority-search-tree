#include <algorithm>	// To use sort()
#include <string>		// To use string-building functions

#include "err-chk.h"

template <typename T>
StaticPSTGPU<T>::StaticPSTGPU(PointStructGPU<T> *const &pt_arr, size_t num_elems)
{
	if (num_elems == 0)
	{
		root_d = nullptr;
		num_elem_slots = 0;
		return;
	}

	// Check number of GPUs attached to machine
	int num_devices = 0;

	gpuErrorCheck(cudaGetDeviceCount(&num_devices), "Error in getting number of devices: ");
	if (num_devices < 1)	// No GPUs attached
		throwErr("Error: " + std::to_string(num_devices) + " GPUs attached to host");

	int device_ind = 0;		// Instantiate outside of for loop so it remains available
	cudaDeviceProp device;
	for (device_ind = 0; device_ind < num_devices; device_ind++)
	{
		gpuErrorCheck(cudaGetDeviceProperties(&device, device_ind),
						"Error in getting device properties of device "
						+ std::to_string(device_ind) + " of " + std::to_string(num_devices)
						+ " total devices: ");

		if (device.unifiedAddressing)
			break;
	}

	if (!device.unifiedAddressing)	// Unified virtual addressing unsupported
		throwErr("Error: none of the " + std::to_string(num_devices)
					+ " devices attached to the system support unified virtual addressing");

	int curr_device;
	gpuErrorCheck(cudaGetDevice(&curr_device), "Error getting default device index: ");

	if (curr_device != device_ind)
		gpuErrorCheck(cudaSetDevice(device_ind), "Error setting default device to device "
						+ std::to_string(device_ind) + " of " + std::to_string(num_devices)
						+ " total devices: ");


	// Save GPU info for later usage
	dev_ind = device_ind;
	dev_warp_size = device.warpSize;
	num_devs = num_devices;

	this->num_elems = num_elems;
	/*
		Minimum number of array slots necessary to construct any complete tree with num_elems elements is 1 less than the smallest power of 2 greater than num_elems
		Tree is fully balanced by construction, with the placement of nodes in the partially empty last row being unknown
	*/
	// Number of element slots in each container subarray is nextGreaterPowerOf2(num_elems) - 1
	num_elem_slots = nextGreaterPowerOf2(num_elems) - 1;

	// Calculate total array size in units of sizeof(T) bytes so that array will satisfy datatype T's alignment requirements (as T is the largest datatype present in this data structure)
	size_t tot_arr_size_num_Ts = calcTotArrSizeNumTs(num_elem_slots);

	/*
		Space needed for instantiation = tree size + num_elems * (size of PointStructGPU + 3 * size of PointStructGPU pointers)
			Enough space to contain 3 size_t indices for every node is needed because the splitting of pointers in the dim2_val array at each node creates a need for the dim2_val arrays to be duplicated
		Space requirement is greater than that needed for reporting nodes, which is simply at most tree_size + num_elems * size of PointStructGPU
	*/
	const size_t global_mem_needed = tot_arr_size_num_Ts * sizeof(T)
									 + num_elems * (sizeof(PointStructGPU<T>) + 3 * sizeof(size_t));
	if (global_mem_needed > device.totalGlobalMem)
		throwErr("Error: needed global memory space of " + std::to_string(global_mem_needed)
					+ " B required for data structure and processing exceeds limit of global memory = "
					+ std::to_string(device.totalGlobalMem) + " B on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs));

	// Allocate as a T array so that alignment requirements for larger data types are obeyed
	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	T *complete_arr = new T[tot_arr_size_num_Ts]();

	if (complete_arr == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems)
					+ " elements of type " + typeid(T).name()
					+ " to instantiate host copy of root array");

	T *complete_arr_d;

	gpuErrorCheck(cudaMalloc(&complete_arr_d, tot_arr_size_num_Ts * sizeof(T)),
					"Error in allocating priority search tree storage array on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");

	gpuErrorCheck(cudaMemcpy(complete_arr_d, complete_arr, tot_arr_size_num_Ts * sizeof(T), cudaMemcpyDefault),
					"Error in zero-initialising on-device priority search tree storage array via cudaMemcpy() on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");

	delete[] complete_arr;

	root_d = complete_arr_d;

	// Create two arrays of PointStructGPU indices for processing PointStructGPU objects on CPU or GPU
	size_t *dim1_val_ind_arr = new size_t[num_elems];
	if (dim1_val_ind_arr == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems)
					+ " elements of type size_t to dim1_val_ind_arr");
	size_t *dim2_val_ind_arr = new size_t[num_elems];
	if (dim2_val_ind_arr == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elems)
					+ " elements of type size_t to dim2_val_ind_arr");

	for (size_t i = 0; i < num_elems; i++)
		dim1_val_ind_arr[i] = dim2_val_ind_arr[i] = i;

	// Sort dimension-1 values index array in ascending order; in-place sort using a curried comparison function
	std::sort(dim1_val_ind_arr, dim1_val_ind_arr + num_elems,
				[](PointStructGPU<T> *const &pt_arr)
					{
						// [&] captures all variables in enclosing scope by reference so that they can be used within the body of the lambda function
						return [&](const size_t &i, const size_t &j)
							{
								return pt_arr[i].compareDim1(pt_arr[j]) < 0;
							};
					}(pt_arr));	// Parentheses immediately after a lambda definition serves to call it with the given parameter

	// Sort dimension-2 values index array in descending order; in-place sort using a curried comparison function
	std::sort(dim2_val_ind_arr, dim2_val_ind_arr + num_elems,
				[](PointStructGPU<T> *const &pt_arr)
					{
						return [&](const size_t &i, const size_t &j)
							{
								return pt_arr[i].compareDim2(pt_arr[j]) > 0;
							};
					}(pt_arr));


	// Create GPU-side array of PointStructGPU indices to store sorted results; as this array is not meant to be permanent, avoid storing the two arrays as one contiguous array in order to avoid allocation failure due to global memory fragmentation
	size_t *dim1_val_ind_arr_d;
	size_t *dim2_val_ind_arr_d;
	size_t *dim2_val_ind_arr_secondary_d;

	// Create GPU-side array of PointStructGPU objects for the index arrays to reference
	PointStructGPU<T> *pt_arr_d;


	gpuErrorCheck(cudaMalloc(&dim1_val_ind_arr_d, num_elems * sizeof(size_t)),
					"Error in allocating array of PointStructGPU indices ordered by dimension 1 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaMalloc(&dim2_val_ind_arr_d, num_elems * sizeof(size_t)),
					"Error in allocating array of PointStructGPU indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaMalloc(&dim2_val_ind_arr_secondary_d, num_elems * sizeof(size_t)),
					"Error in allocating secondary array of PointStructGPU indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaMalloc(&pt_arr_d, num_elems * sizeof(PointStructGPU<T>)),
					"Error in allocating array of PointStructGPU objects on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");


	gpuErrorCheck(cudaMemcpy(dim1_val_ind_arr_d, dim1_val_ind_arr, num_elems * sizeof(size_t), cudaMemcpyDefault),
					"Error in copying array of PointStructGPU indices ordered by dimension 1 to device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaMemcpy(dim2_val_ind_arr_d, dim2_val_ind_arr, num_elems * sizeof(size_t), cudaMemcpyDefault), 
					"Error in copying array of PointStructGPU indices ordered by dimension 2 to device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaMemcpy(pt_arr_d, pt_arr, num_elems * sizeof(PointStructGPU<T>), cudaMemcpyDefault), 
					"Error in copying array of PointStructGPU objects to device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");


	delete[] dim1_val_ind_arr;
	delete[] dim2_val_ind_arr;

	// Populate tree with a number of threads that is a multiple of the warp size
	populateTree<<<1, dev_warp_size, dev_warp_size * sizeof(size_t) * 3>>>
				(root_d, num_elem_slots, pt_arr_d, dim1_val_ind_arr_d, dim2_val_ind_arr_d, dim2_val_ind_arr_secondary_d, 0, num_elems, 0);


	gpuErrorCheck(cudaDeviceSynchronize(), "Error in synchronizing with device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ " after tree construction: ");

	// All threads have finished using these arrays; free them and return
	gpuErrorCheck(cudaFree(dim1_val_ind_arr_d),
					"Error in freeing array of PointStructGPU indices ordered by dimension 1 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaFree(dim2_val_ind_arr_d),
					"Error in freeing array of PointStructGPU indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaFree(dim2_val_ind_arr_secondary_d), 
					"Error in freeing secondary array of PointStructGPU indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaFree(pt_arr_d),
					"Error in freeing array of PointStructGPU objects on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
}

// static keyword should only be used when declaring a function in the header file
template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::constructNode(T *const &root_d,
																const size_t &num_elem_slots,
																PointStructGPU<T> *const &pt_arr_d,
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
																size_t &right_subarr_num_elems)
{
	size_t median_dim1_val_ind;

	// Treat dim1_val_ind_arr_ind[max_dim2_val_dim1_array_ind] as a removed element (but don't actually remove the element for performance reasons

	if (num_subelems_arr[threadIdx.x] == 1)		// Base case
	{
		median_dim1_val_ind = subelems_start_inds_arr[threadIdx.x];
		left_subarr_num_elems = 0;
		right_subarr_num_elems = 0;
	}
	// max_dim2_val originally comes from the part of the array to the left of median_dim1_val
	else if (max_dim2_val_dim1_array_ind < subelems_start_inds_arr[threadIdx.x] + num_subelems_arr[threadIdx.x]/2)
	{
		// As median values are always put in the left subtree, when the subroot value comes from the left subarray, the median index is given by num_elems/2, which evenly splits the array if there are an even number of elements remaining or makes the left subtree larger by one element if there are an odd number of elements remaining
		median_dim1_val_ind = subelems_start_inds_arr[threadIdx.x]
								+ num_subelems_arr[threadIdx.x]/2;
		// max_dim2_val has been removed from the left subarray, so there are median_dim1_val_ind elements remaining on the left side
		left_subarr_num_elems = median_dim1_val_ind - subelems_start_inds_arr[threadIdx.x];
		right_subarr_num_elems = subelems_start_inds_arr[threadIdx.x] + num_subelems_arr[threadIdx.x]
									- median_dim1_val_ind - 1;
	}
	/*
		max_dim2_val originally comes from the part of the array to the right of median_dim1_val, i.e.
			max_dim2_val_dim1_array_ind >= subelems_start_inds_arr[threadIdx.x] + num_subelems_arr[threadIdx.x]/2
	*/
	else
	{
		median_dim1_val_ind = subelems_start_inds_arr[threadIdx.x]
								+ num_subelems_arr[threadIdx.x]/2 - 1;

		left_subarr_num_elems = median_dim1_val_ind - subelems_start_inds_arr[threadIdx.x] + 1;
		right_subarr_num_elems = subelems_start_inds_arr[threadIdx.x] + num_subelems_arr[threadIdx.x]
									- median_dim1_val_ind - 2;
	}

	setNode(root_d, target_node_ind, num_elem_slots,
			pt_arr_d[dim2_val_ind_arr_d[subelems_start_inds_arr[threadIdx.x]]],
			pt_arr_d[dim1_val_ind_arr_d[median_dim1_val_ind]].dim1_val);

	if (left_subarr_num_elems > 0)
	{
		TreeNode::setLeftChild(getBitcodesRoot(root_d, num_elem_slots), target_node_ind);

		// Always place median value in left subtree
		// max_dim2_val is to the left of median_dim1_val in dim1_val_ind_arr_d; shift all entries up to median_dim1_val_ind leftward, overwriting max_dim2_val_dim1_array_ind
		if (max_dim2_val_dim1_array_ind < median_dim1_val_ind)
			// memcpy() is undefined if the source and destination regions overlap, and the safe memmove() (that behaves as if the source values were first copied to an intermediate array) does not exist on CUDA within kernel code
			for (size_t i = max_dim2_val_dim1_array_ind; i < median_dim1_val_ind; i++)
				dim1_val_ind_arr_d[i] = dim1_val_ind_arr_d[i+1];
		// Otherwise, max_dim2_val is to the right of median_dim1_val_ind in dim1_val_ind_arr_d; leave left subarray as is
	}
	if (right_subarr_num_elems > 0)
	{
		TreeNode::setRightChild(getBitcodesRoot(root_d, num_elem_slots), target_node_ind);

		// max_dim2_val is to the right of median_dim1_val_ind in dim1_val_ind_arr_d; shift all entries after max_dim2_val_dim1_array_ind leftward, overwriting max_dim2_val_array_ind
		if (max_dim2_val_dim1_array_ind > median_dim1_val_ind)
			for (size_t i = max_dim2_val_dim1_array_ind;
					i < subelems_start_inds_arr[threadIdx.x] + num_subelems_arr[threadIdx.x] - 1;
					i++)
				dim1_val_ind_arr_d[i] = dim1_val_ind_arr_d[i+1];
		// Otherwise, max_dim2_val is to the left of median_dim1_val in dim1_val_ind_arr_d; leave right subarray as is
	}


	// Choose median_dim1_val_ind + 1 as the starting index for all right subarrays, as this is the only index that is valid no matter whether max_dim2_val is to the left or right of med_dim1_val
	right_subarr_start_ind = median_dim1_val_ind + 1;


	// Iterate through dim2_val_ind_arr_d, placing data with lower dimension-1 value in the subarray for the left subtree and data with higher dimension-1 value in the subarray for the right subtree
	// Note that because subarrays' sizes were allocated based on this same data, there should not be a need to check that left_dim2_subarr_iter_ind < left_subarr_num_elems (and similarly for the right side)
	if (TreeNode::hasChildren(getBitcodesRoot(root_d, num_elem_slots)[target_node_ind]))
	{
		size_t left_dim2_subarr_iter_ind = subelems_start_inds_arr[threadIdx.x];
		size_t right_dim2_subarr_iter_ind = right_subarr_start_ind;

		// Skip over first (largest) element in dim2_val_ind_arr_d, as it has been already placed in the current node
		for (size_t i = subelems_start_inds_arr[threadIdx.x] + 1;
				i < subelems_start_inds_arr[threadIdx.x] + num_subelems_arr[threadIdx.x];
				i++)
		{
			// dim2_val_ind_arr_d[i] is the index of a PointStructGPU that comes before or is the PointStructGPU of median dim1 value in dim1_val_ind_arr_d
			if (pt_arr_d[dim2_val_ind_arr_d[i]].compareDim1(pt_arr_d[dim1_val_ind_arr_d[median_dim1_val_ind]]) <= 0)
				// Postfix ++ returns the current value before incrementing
				dim2_val_ind_arr_secondary_d[left_dim2_subarr_iter_ind++] = dim2_val_ind_arr_d[i];
			// dim2_val_ind_arr_d[i] is the index of a PointStructGPU that comes after the PointStructGPU of median dim1 value in dim1_val_ind_arr_d
			else
				dim2_val_ind_arr_secondary_d[right_dim2_subarr_iter_ind++] = dim2_val_ind_arr_d[i];
		}
	}
}

template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::splitLeftSearchWork(T *const &root_d, const size_t &num_elem_slots, const size_t &target_node_ind, PointStructGPU<T> *const res_pt_arr_d, const T &max_dim1_val, const T &min_dim2_val, long long *const &search_inds_arr, unsigned char *const &search_codes_arr)
{
	// Find next inactive thread by iterating through search_inds_arr atomically
	// i < blockDim.x check comes before atomicCAS() operation because short-circuit evaluation will ensure atomicCAS() does not write to a point beyond the end of search_inds_arr
	// atomicCAS(addr, cmp, val) takes the value old := *addr, sets *addr = (old == cmp ? val : old) and returns old; the swap took place iff the return value old == cmp; all calculations are done as one atomic operation
	// Casting necessary to satisfy atomicCAS()'s signature of unsigned long long
	size_t i;
	for (i = 0; i < blockDim.x
			&& static_cast<long long>(atomicCAS(reinterpret_cast<unsigned long long *>(search_inds_arr + i),
												static_cast<unsigned long long>(INACTIVE_IND),
												target_node_ind))
										!= INACTIVE_IND;
			i++)
		{}
	// Upon exit, i either contains the index of the thread that will report all nodes in the corresponding subtree; or i >= blockDim.x
	if (i >= blockDim.x)	// No inactive threads; use dynamic parallelism
	{
		// report-all searches never become normal searches again, so do not need shared memory for a search_codes_arr, just a search_inds_arr
		twoSidedLeftSearchGlobal<<<1, blockDim.x, blockDim.x * (sizeof(long long) + sizeof(unsigned char))>>>
			(root_d, num_elem_slots, target_node_ind, res_pt_arr_d, max_dim1_val, min_dim2_val);
	}
	else	// Inactive thread has ID i
	{
		search_inds_arr[i] = target_node_ind;
		search_codes_arr[i] = LEFT_SEARCH;
	}
}

template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::do3SidedSearchDelegation(const unsigned char &curr_node_bitcode, T *const &root_d, const size_t &num_elem_slots, PointStructGPU<T> *const res_pt_arr_d, const T &min_dim1_val, const T &max_dim1_val, const T &curr_node_med_dim1_val, const T &min_dim2_val, long long &search_ind, long long *const &search_inds_arr, unsigned char &search_code, unsigned char *const &search_codes_arr)
{
	// Splitting of query is only possible if the current node has two children and min_dim1_val <= curr_node_med_dim1_val <= max_dim1_val; the equality on max_dim1_val is for the edge case where a median point may be duplicated, with one copy going to the left subtree and the other to the right subtree
	if (min_dim1_val <= curr_node_med_dim1_val
			&& curr_node_med_dim1_val <= max_dim1_val)
	{
		// Query splits over median and node has two children; split into 2 two-sided queries
		if (TreeNode::hasLeftChild(curr_node_bitcode)
				&& TreeNode::hasRightChild(curr_node_bitcode))
		{
			// Delegate work of searching right subtree to another thread and/or block
			splitLeftSearchWork(root_d, num_elem_slots, TreeNode::getRightChild(search_ind),
									res_pt_arr_d, max_dim1_val, min_dim2_val,
									search_inds_arr, search_codes_arr);

			// Prepare to search left subtree with a two-sided right search in the next iteration
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = SearchCodes::RIGHT_SEARCH;
		}
		// No right child, so perform a two-sided right query on the left child
		else if (TreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = SearchCodes::RIGHT_SEARCH;
		}
		// No left child, so perform a two-sided left query on the right child
		else
		{
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getRightChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = SearchCodes::LEFT_SEARCH;
		}
	}
	// Perform three-sided search on left child
	else if (max_dim1_val < curr_node_med_dim1_val
				&& TreeNode::hasLeftChild(curr_node_bitcode))
	{
		// Search code is already a THREE_SEARCH
		search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
	}
	// Perform three-sided search on right child
	// Only remaining possibility, as all others mean the thread is inactive:
	//		curr_node_med_dim1_val < min_dim1_val && TreeNode::hasRightChild(curr_node_bitcode)
	else
	{
		// Search code is already a THREE_SEARCH
		search_inds_arr[threadIdx.x] = search_ind = TreeNode::getRightChild(search_ind);
	}
}

template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::doLeftSearchDelegation(const bool range_split_poss, const unsigned char &curr_node_bitcode, T *const &root_d, const size_t &num_elem_slots, PointStructGPU<T> *const res_pt_arr_d, const T &min_dim2_val, long long &search_ind, long long *const &search_inds_arr, unsigned char &search_code, unsigned char *const &search_codes_arr)
{
	// Report all nodes in left subtree, "recurse" search on right
	// Because reportAllNodesGlobal uses less shared memory, prefer launching reportAllNodesGlobal over launching a search when utilising dynamic parallelism
	// Though the upper bound of the dimension-1 search range is typically open, if there are duplicates of the median point and one happens to be allocated to each subtree, both trees must be traversed for correctness
	if (range_split_poss)
	{
		// If current node has two children, parallelise search at each child
		if (TreeNode::hasLeftChild(curr_node_bitcode)
				&& TreeNode::hasRightChild(curr_node_bitcode))
		{
			// Delegate work of reporting all nodes in left child to another thread and/or block
			splitReportAllNodesWork(root_d, num_elem_slots, TreeNode::getLeftChild(search_ind),
										res_pt_arr_d, min_dim2_val,
										search_inds_arr, search_codes_arr);


			// Prepare to search right subtree in the next iteration
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getRightChild(search_ind);
		}
		// Node only has a left child; report all on left child
		else if (TreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = REPORT_ALL;
		}
		// Node only has a right child; search on right child
		else if (TreeNode::hasRightChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getRightChild(search_ind);
		}
	}
	// !split_range_poss
	// Only left subtree can possibly contain valid entries; search left subtree
	else if (TreeNode::hasLeftChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
	}
}

template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::doRightSearchDelegation(const bool range_split_poss, const unsigned char &curr_node_bitcode, T *const &root_d, const size_t &num_elem_slots, PointStructGPU<T> *const res_pt_arr_d, const T &min_dim2_val, long long &search_ind, long long *const &search_inds_arr, unsigned char &search_code, unsigned char *const &search_codes_arr)
{
	// Report all nodes in right subtree, "recurse" search on left
	// Because reportAllNodesGlobal uses less shared memory, prefer launching reportAllNodesGlobal over launching a search when utilising dynamic parallelism
	if (range_split_poss)
	{
		// If current node has two children, parallelise search at each child
		if (TreeNode::hasLeftChild(curr_node_bitcode)
				&& TreeNode::hasRightChild(curr_node_bitcode))
		{
			// Delegate work of reporting all nodes in right child to another thread and/or block
			splitReportAllNodesWork(root_d, num_elem_slots, TreeNode::getRightChild(search_ind),
										res_pt_arr_d, min_dim2_val,
										search_inds_arr, search_codes_arr);

			// Continue search in the next iteration
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
		}
		// Node only has a right child; report all on right child
		else if (TreeNode::hasRightChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getRightChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = REPORT_ALL;
		}
		// Node only has a left child; search on left child
		else if (TreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
		}
	}
	// !range_split_poss
	// Only right subtree can possibly contain valid entries; search right subtree
	else if (TreeNode::hasRightChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = TreeNode::getRightChild(search_ind);
	}
}

// C++ allows trailing arguments to have default values; need not specify the default arguments after their initial declaration
template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::doReportAllNodesDelegation(const unsigned char &curr_node_bitcode, T *const &root_d, const size_t &num_elem_slots, PointStructGPU<T> *const res_pt_arr_d, const T &min_dim2_val, long long &search_ind, long long *const &search_inds_arr, unsigned char *const &search_codes_arr)
{
	if (TreeNode::hasLeftChild(curr_node_bitcode)
			&& TreeNode::hasRightChild(curr_node_bitcode))
	{
		// Delegate reporting of all nodes in right child to another thread and/or block
		splitReportAllNodesWork(root_d, num_elem_slots, TreeNode::getRightChild(search_ind),
									res_pt_arr_d, min_dim2_val,
									search_inds_arr, search_codes_arr);

		// Prepare for next iteration; because execution is already in this branch, search_codes_arr[threadIdx.x] == REPORT_ALL already
		search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
	}
	// Node only has a left child; report all on left child
	else if (TreeNode::hasLeftChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = TreeNode::getLeftChild(search_ind);
	}
	// Node only has a right child; report all on right child
	else if (TreeNode::hasRightChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = TreeNode::getRightChild(search_ind);
	}
}

template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::splitReportAllNodesWork(T *const &root_d, const size_t &num_elem_slots, const size_t &target_node_ind, PointStructGPU<T> *const res_pt_arr_d, const T &min_dim2_val, long long *const &search_inds_arr, unsigned char *const &search_codes_arr)
{
	// Find next inactive thread by iterating through search_inds_arr atomically
	// i < blockDim.x check comes before atomicCAS() operation because short-circuit evaluation will ensure atomicCAS() does not write to a point beyond the end of search_inds_arr
	// atomicCAS(addr, cmp, val) takes the value old := *addr, sets *addr = (old == cmp ? val : old) and returns old; the swap took place iff the return value old == cmp; all calculations are done as one atomic operation
	// Casting necessary to satisfy atomicCAS()'s signature of unsigned long long
	size_t i;
	for (i = 0; i < blockDim.x
			&& static_cast<long long>(atomicCAS(reinterpret_cast<unsigned long long *>(search_inds_arr + i),
												static_cast<unsigned long long>(INACTIVE_IND),
												target_node_ind))
										!= INACTIVE_IND;
			i++)
		{}
	// Upon exit, i either contains the index of the thread that will report all nodes in the corresponding subtree; or i >= blockDim.x
	if (i >= blockDim.x)	// No inactive threads; use dynamic parallelism
	{
		// report-all searches never become normal searches again, so do not need shared memory for a search_codes_arr, just a search_inds_arr
		reportAllNodesGlobal<<<1, blockDim.x, blockDim.x * sizeof(long long)>>>
			(root_d, num_elem_slots, target_node_ind, res_pt_arr_d, min_dim2_val);
	}
	else	// Inactive thread has ID i
	{
		search_inds_arr[i] = target_node_ind;

		// For applicability to splitting of work when called from reportAllNodesGlobal()
		if (search_codes_arr != nullptr)
			search_codes_arr[i] = REPORT_ALL;
	}
}

template <typename T>
__forceinline__ __device__ void StaticPSTGPU<T>::detInactivity(long long &search_ind, long long *const &search_inds_arr, bool &cont_iter, unsigned char *const search_code_ptr, unsigned char *const &search_codes_arr)
{
	// INACTIVE threads check wehther they should be active in the next iteration; if not, and all threads are inactive, set iteration toggle to false

	// Thread has been assigned work; update local variables accordingly
	if (search_ind == INACTIVE_IND
			&& search_inds_arr[threadIdx.x] != INACTIVE_IND)
	{
		search_ind = search_inds_arr[threadIdx.x];
		// These two should always have or not have nullptr value at the same time, but add this safeguard just in case
		if (search_code_ptr != nullptr && search_codes_arr != nullptr)
			*search_code_ptr = search_codes_arr[threadIdx.x];
	}
	// Thread remains inactive; check if all other threads are inactive; if so, all processing has completed
	else if (search_ind == INACTIVE_IND)
	{
		int i = 0;
		for (i = 0; i < blockDim.x; i++)
			if (search_inds_arr[i] != INACTIVE_IND)
				break;
		if (i >= blockDim.x)	// No threads are active; all processing has completed
			cont_iter = false;
	}
}

template <typename T>
__forceinline__ size_t StaticPSTGPU<T>::calcTotArrSizeNumTs(const size_t num_elem_slots)
{
	/*
		tot_arr_size_num_Ts = ceil(num_elem_slots * (num_val_subarrs + num_Ts/bitcode))
							= ceil(num_elem_slots * (num_val_subarrs + 1 B/bitcode * 1 T/#Bs))
							= ceil(num_elem_slots * (num_val_subarrs + 1/sizeof(T)))
							= ceil(num_elem_slots * num_val_subarrs + num_elem_slots / sizeof(T))
							= num_elem_slots * num_val_subarrs + ceil(num_elem_slots / sizeof(T))
			With integer truncation:
				if num_elem_slots % codes_per_byte != 0:
							= num_elem_slots * num_val_subarrs + num_elem_slots / sizeof(T) + 1
				if num_elem_slots % codes_per_byte == 0:
							= num_elem_slots * num_val_subarrs + num_elem_slots / sizeof(T)
	*/
	size_t tot_arr_size_num_Ts = num_val_subarrs * num_elem_slots + num_elem_slots/sizeof(T);
	if (num_elem_slots % sizeof(T) != 0)
		tot_arr_size_num_Ts++;
	return tot_arr_size_num_Ts;
}

template <typename T>
__forceinline__ __host__ __device__ size_t StaticPSTGPU<T>::expOfNextGreaterPowerOf2(const size_t num)
{
	/*
		Smallest power of 2 greater than num is equal to 2^ceil(lg(num + 1))
		ceil(lg(num + 1)) is equal to the number of right bitshifts necessary to make num = 0 (after integer truncation); this method of calcalation is used in order to prevent imprecision of float conversion from causing excessively large (and therefore incorrect) returned integer values
	*/
	unsigned exp = 0;
	while (num >> exp != 0)
		exp++;
	return exp;
}

template <typename T>
__forceinline__ __host__ __device__ long long StaticPSTGPU<T>::binarySearch(PointStructGPU<T> *const &pt_arr, size_t *const &dim1_val_ind_arr, PointStructGPU<T> &elem_to_find, const size_t &init_ind, const size_t &num_elems)
{
	size_t low_ind = init_ind;
	size_t high_ind = init_ind + num_elems;
	size_t mid_ind;		// Avoid reinstantiating mid_ind in every iteration
	// Search is done in the range [low_ind, high_ind)
	while (low_ind < high_ind)
	{
		mid_ind = (low_ind + high_ind)/2;
		// Location in dim1_val_ind_arr of elem_to_find has been found
		if (pt_arr[dim1_val_ind_arr[mid_ind]] == elem_to_find
			&& pt_arr[dim1_val_ind_arr[mid_ind]].comparisonTiebreaker(elem_to_find) == 0)
			return mid_ind;
		// elem_to_find is before middle element; recurse on left subarray
		else if (elem_to_find.compareDim1(pt_arr[dim1_val_ind_arr[mid_ind]]) < 0)
			high_ind = mid_ind;
		// elem_to_find is after middle element; recurse on right subarray
		else	// elem_to_find.compareDim1(pt_arr[dim1_val_ind_arr[mid_ind]]) > 0
			low_ind = mid_ind + 1;
	}
	return -1;	// Element not found
}

// const keyword after method name indicates that the method does not modify any data members of the associated class
template <typename T>
void StaticPSTGPU<T>::print(std::ostream &os) const
{
	if (num_elem_slots == 0)
	{
		os << "Tree is empty\n";
		return;
	}
	T *temp_root = new T[calcTotArrSizeNumTs(num_elem_slots)];
	if (temp_root == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elem_slots)
					+ " elements of type " + typeid(T).name() + "to temp_root");
	gpuErrorCheck(cudaMemcpy(temp_root, root_d, calcTotArrSizeNumTs(num_elem_slots) * sizeof(T), cudaMemcpyDefault),
					"Error in copying array underlying StaticPSTGPU instance from device to host: ");

	std::string prefix = "";
	std::string child_prefix = "";
	printRecur(os, temp_root, 0, num_elem_slots, prefix, child_prefix);

	delete[] temp_root;
}

template <typename T>
void StaticPSTGPU<T>::printRecur(std::ostream &os, T *const &tree_root, const size_t curr_ind, const size_t num_elem_slots, std::string prefix, std::string child_prefix) const
{
	os << prefix << '(' << getDim1ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< ", " << getDim2ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< "; " << getMedDim1ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< ')';
	const unsigned char curr_node_bitcode = getBitcodesRoot(tree_root, num_elem_slots)[curr_ind];
	if (TreeNode::hasLeftChild(curr_node_bitcode)
			&& TreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, TreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "├─(R)─ ", child_prefix + "│      ");
		printRecur(os, tree_root, TreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
	else if (TreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, TreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(R)─ ", child_prefix + "       ");
	}
	else if (TreeNode::hasLeftChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, TreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
}
