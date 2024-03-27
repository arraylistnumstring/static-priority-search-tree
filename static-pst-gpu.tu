#include <string>				// To use string-building functions
#include <thrust/execution_policy.h>	// To use thrust::device execution policy for sorting
#include <thrust/sort.h>		// To use parallel sorting algorithm

#include "err-chk.h"
#include "helper-cuda--modified.h"

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::StaticPSTGPU(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr, size_t num_elems)
{
	if (num_elems == 0)
	{
		root_d = nullptr;
		num_elem_slots = 0;
		return;
	}

	// Check and save number of GPUs attached to machine
	gpuErrorCheck(cudaGetDeviceCount(&num_devs), "Error in getting number of devices: ");
	if (num_devs < 1)	// No GPUs attached
		throwErr("Error: " + std::to_string(num_devs) + " GPUs attached to host");

	// Use modified version of CUDA's gpuGetMaxGflopsDeviceId() to get top-performing GPU capable of unified virtual addressing; also used so that device in use is the same as that for marching cubes
	dev_ind = gpuGetMaxGflopsDeviceId();
	gpuErrorCheck(cudaGetDeviceProperties(&dev_props, dev_ind),
					"Error in getting device properties of device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ " total devices: ");

	gpuErrorCheck(cudaSetDevice(dev_ind), "Error setting default device to device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ " total devices: ");

	this->num_elems = num_elems;
	/*
		Minimum number of array slots necessary to construct any complete tree with num_elems elements is 1 less than the smallest power of 2 greater than num_elems
		Tree is fully balanced by construction, with the placement of nodes in the partially empty last row being unknown
	*/
	// Number of element slots in each container subarray is nextGreaterPowerOf2(num_elems) - 1
	num_elem_slots = nextGreaterPowerOf2(num_elems) - 1;

	// Allocate as a T array so that alignment requirements for larger data types are obeyed
	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	// constexpr if is a C++17 feature that only compiles the branch of code that evaluates to true at compile-time, saving executable space and execution runtime
	size_t tot_arr_size_num_datatype;
	size_t global_mem_needed;
	if constexpr (num_IDs == 0 || sizeof(T) >= sizeof(IDType))
	{
		// No IDs present or sizeof(T) >= sizeof(IDType), so calculate total array size in units of sizeof(T) so that datatype T's alignment requirements will be satisfied
		tot_arr_size_num_datatype = calcTotArrSizeNumUs<T, num_val_subarrs, IDType, num_ID_subarrs>(num_elem_slots);
		global_mem_needed = tot_arr_size_num_datatype * sizeof(T);
	}
	else
	{
		// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
		tot_arr_size_num_datatype = calcTotArrSizeNumUs<IDType, num_ID_subarrs, T, num_val_subarrs>(num_elem_slots);
		global_mem_needed = tot_arr_size_num_datatype * sizeof(IDType);
	}

	/*
		Space needed for instantiation = tree size + num_elems * (size of PointStructTemplate<T, IDType, num_IDs> + 3 * size of PointStructTemplate indices)
			Enough space to contain 3 size_t indices for every node is needed because the splitting of pointers in the dim2_val array at each node creates a need for the dim2_val arrays to be duplicated
		Space requirement is greater than that needed for reporting nodes, which is simply at most tree_size + num_elems * size of PointStructTemplate
	*/
	const size_t num_working_ind_arrays = 3;
	global_mem_needed += num_elems * (sizeof(PointStructTemplate<T, IDType, num_IDs>) + num_working_ind_arrays * sizeof(size_t));
	if (global_mem_needed > device.totalGlobalMem)
		throwErr("Error: needed global memory space of " + std::to_string(global_mem_needed)
					+ " B required for data structure and processing exceeds limit of global memory = "
					+ std::to_string(device.totalGlobalMem) + " B on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs));

	// Memory transfer only permitted for on-host pinned (page-locked) memory, so do operations in the default stream
	if constexpr (num_IDs == 0 || sizeof(T) >= sizeof(IDType))
	{
		// Allocate as a T array so that alignment requirements for larger data types are obeyed
		gpuErrorCheck(cudaMalloc(&root_d, tot_arr_size_num_datatype * sizeof(T)),
						"Error in allocating priority search tree storage array on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ ": ");
	}
	else
	{
		// Allocate as an IDType array so that alignment requirements for larger data types are obeyed
		gpuErrorCheck(cudaMalloc(&root_d, tot_arr_size_num_datatype * sizeof(IDType)),
						"Error in allocating priority search tree storage array on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ ": ");
	}


	// Create GPU-side array of PointStructTemplate<T, IDType, num_IDs> indices to store sorted results; as this array is not meant to be permanent, avoid storing the two arrays as one contiguous array in order to avoid allocation failure due to global memory fragmentation
	size_t *dim1_val_ind_arr_d;
	size_t *dim2_val_ind_arr_d;
	size_t *dim2_val_ind_arr_secondary_d;

	// Create GPU-side array of PointStructTemplate<T, IDType, num_IDs> objects for the index arrays to reference
	PointStructTemplate<T, IDType, num_IDs> *pt_arr_d;


	gpuErrorCheck(cudaMalloc(&dim1_val_ind_arr_d, num_elems * sizeof(size_t)),
					"Error in allocating array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 1 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaMalloc(&dim2_val_ind_arr_d, num_elems * sizeof(size_t)),
					"Error in allocating array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaMalloc(&dim2_val_ind_arr_secondary_d, num_elems * sizeof(size_t)),
					"Error in allocating secondary array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");

	CUcontext *ptr_info;
	gpuErrorCheck(cuPointerGetAttribute(ptr_info, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pt_arr),
					"Error in attempting to get attribute information of pt_arr on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	if (ptr_info == CU_MEMORYTYPE_HOST)		// pt_arr is on host; allocate memory for and copy pt_arr
	{
		gpuErrorCheck(cudaMalloc(&pt_arr_d, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>)),
						"Error in allocating array of PointStructTemplate<T, IDType, num_IDs> objects on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ ": ");

		// Use cudaMemcpy() call as implicit call to cudaDeviceSynchronize(), as cudaMemcpy() calls are only truly asynchronous for pinned memory
		gpuErrorCheck(cudaMemcpy(pt_arr_d, pt_arr, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>), cudaMemcpyDefault), 
						"Error in copying array of PointStructTemplate<T, IDType, num_IDs> objects to device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ ": ");
	}
	else if (ptr_info == CU_MEMORYTYPE_DEVICE)	// pt_arr is already on device; use it as pt_arr_d
	{
		pt_arr_d = pt_arr;
		// No implicit synchronization call as with cudaMemcpy, so instead all cudaDeviceSynchornize() explicitly
		gpuErrorCheck(cudaDeviceSynchronize(), "Error in synchronizing with GPU after "
						+ "allocating data on device " + std::to_string(dev_ind) + " of "
						+ std::to_string(num_devs) + ": ");
	}
	else	// Something has gone very wrong; exit
		return;


	if constexpr (num_IDs == 0 || sizeof(T) >= sizeof(IDType))
	{
		gpuErrorCheck(cudaMemsetAsync(root_d, 0, tot_arr_size_num_data * sizeof(T), cudaStreamFireAndForget),
						"Error in zero-intialising on-device priority search tree storage array via cudaMemset() on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ ": ");
	}
	else
	{
		gpuErrorCheck(cudaMemsetAsync(root_d, 0, tot_arr_size_num_data * sizeof(IDType), cudaStreamFireAndForget),
						"Error in zero-intialising on-device priority search tree storage array via cudaMemset() on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ ": ");
	}

	const size_t index_assign_threads_per_block = 8 * dev_props.warpSize;
	const size_t index_assign_num_blocks = std::min(num_elems % index_assign_threads_per_block == 0 ? num_elems/index_assign_threads_per_block : num_elems/index_assign_threads_per_block + 1, dev_props.warpSize * dev_props.warpSize);

	// Create concurrent streams for index-initialising and sorting the dimension-1 and dimension-2 index arrays
	cudaStream_t stream_dim1;
	gpuErrorCheck(cudaStreamCreateWithFlags(&stream_dim1, cudaStreamNonBlocking),
					"Error in creating asynchronous stream for assignment and sorting of "
					+ "indices by dimension 1 on device " + std::to_string(dev_ind) + " of "
					+ std::to_string(num_devs) + ": ");
	indexAssignment<<<index_assign_num_blocks, index_assign_threads_per_block, 0, stream_dim1>>>(dim1_val_ind_arr_d, num_elems);
	// Sort dimension-1 values index array in ascending order; in-place sort using a curried comparison function; guaranteed O(n) running time or better
	// Execution policy of thrust::cuda::par.on(stream_dim1) guarantees kernel is submitted to stream_dim1
	thrust::sort(thrust::cuda::par.on(stream_dim1), dim1_val_ind_arr_d, dim1_val_ind_arr_d + num_elems,
					Dim1ValIndCompIncOrd(pt_arr_d));
	// cudaStreamDestroy() is also a kernel submitted to the indicated stream, so it only runs once all previous calls have completed
	gpuErrorCheck(cudaStreamDestroy(stream_dim1), "Error in destroying asynchronous stream for "
					+ "assignment and sorting of indices by dimension 1 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	
	cudaStream_t stream_dim2;
	gpuErrorCheck(cudaStreamCreateWithFlags(&stream_dim2, cudaStreamNonBlocking),
					"Error in creating asynchronous stream for assignment and sorting of "
					+ "indices by dimension 2 on device " + std::to_string(dev_ind) + " of "
					+ std::to_string(num_devs) + ": ");
	indexAssignment<<<index_assign_num_blocks, index_assign_threads_per_block, 0, stream_dim2>>>(dim2_val_ind_arr_d, num_elems);
	// Sort dimension-2 values index array in descending order; in-place sort using a curried comparison function; guaranteed O(n) running time or better
	thrust::sort(thrust::cuda::par.on(stream_dim2), dim2_val_ind_arr_d, dim2_val_ind_arr_d + num_elems,
					Dim2ValIndCompDecOrd(pt_arr_d));
	gpuErrorCheck(cudaStreamDestroy(stream_dim2), "Error in destroying asynchronous stream for "
					+ "assignment and sorting of indices by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");


	// For correctness, must wait for all streams doing pre-construction pre-processing work to complete before continuing
	gpuErrorCheck(cudaDeviceSynchronize(), "Error in synchronizing with device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ " after tree pre-construction pre-processing: ");

	// Populate tree with a number of threads that is a multiple of the warp size
	populateTree<<<1, dev_props.warpSize, dev_props.warpSize * sizeof(size_t) * num_working_ind_arrays>>>
				(root_d, num_elem_slots, pt_arr_d, dim1_val_ind_arr_d, dim2_val_ind_arr_d, dim2_val_ind_arr_secondary_d, 0, num_elems, 0);


	gpuErrorCheck(cudaDeviceSynchronize(), "Error in synchronizing with device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ " after tree construction: ");

	// All threads have finished using these arrays; free them and return
	gpuErrorCheck(cudaFree(dim1_val_ind_arr_d),
					"Error in freeing array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 1 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaFree(dim2_val_ind_arr_d),
					"Error in freeing array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaFree(dim2_val_ind_arr_secondary_d), 
					"Error in freeing secondary array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
	gpuErrorCheck(cudaFree(pt_arr_d),
					"Error in freeing array of PointStructTemplate<T, IDType, num_IDs> objects on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ ": ");
}

// const keyword after method name indicates that the method does not modify any data members of the associated class
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
	if (num_elem_slots == 0)
	{
		os << "Tree is empty\n";
		return;
	}

	size_t tot_arr_size_num_datatype;
	T *temp_root;
	if constexpr (num_IDs == 0 || sizeof(T) >= sizeof(IDType))
	{
		// No IDs present or sizeof(T) >= sizeof(IDType), so calculate total array size in units of sizeof(T) so that datatype T's alignment requirements will be satisfied
		tot_arr_size_num_datatype = calcTotArrSizeNumUs<T, num_val_subarrs, IDType, num_ID_subarrs>(num_elem_slots);
		temp_root = new T[tot_arr_size_num_Ts]();
	}
	else
	{
		// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
		tot_arr_size_num_datatypes = calcTotArrSizeNumUs<IDType, num_ID_subarrs, T, num_val_subarrs>(num_elem_slots);
		temp_root = reinterpret_cast<T *>(new IDType[tot_arr_size_num_IDTypes]());
	}
	
	if (temp_root == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elem_slots)
					+ " elements of type " + typeid(T).name() + "to temp_root");

	if constexpr (num_IDs == 0 || sizeof(T) >= sizeof(IDType))
	{
		gpuErrorCheck(cudaMemcpy(temp_root, root_d, tot_arr_size_num_datatypes * sizeof(T), cudaMemcpyDefault),
						"Error in copying array underlying StaticPSTGPU instance from device to host: ");
	}
	else
	{
		gpuErrorCheck(cudaMemcpy(temp_root, root_d, tot_arr_size_num_datatypes * sizeof(IDType), cudaMemcpyDefault),
						"Error in copying array underlying StaticPSTGPU instance from device to host: ");
	}

	std::string prefix = "";
	std::string child_prefix = "";
	printRecur(os, temp_root, 0, num_elem_slots, prefix, child_prefix);

	delete[] temp_root;
}

// static keyword should only be used when declaring a function in the header file
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::constructNode(T *const &root_d,
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
			// dim2_val_ind_arr_d[i] is the index of a PointStructTemplate<T, IDType, num_IDs> that comes before or is the PointStructTemplate<T, IDType, num_IDs> of median dim1 value in dim1_val_ind_arr_d
			if (pt_arr_d[dim2_val_ind_arr_d[i]].compareDim1(pt_arr_d[dim1_val_ind_arr_d[median_dim1_val_ind]]) <= 0)
				// Postfix ++ returns the current value before incrementing
				dim2_val_ind_arr_secondary_d[left_dim2_subarr_iter_ind++] = dim2_val_ind_arr_d[i];
			// dim2_val_ind_arr_d[i] is the index of a PointStructTemplate<T, IDType, num_IDs> that comes after the PointStructTemplate<T, IDType, num_IDs> of median dim1 value in dim1_val_ind_arr_d
			else
				dim2_val_ind_arr_secondary_d[right_dim2_subarr_iter_ind++] = dim2_val_ind_arr_d[i];
		}
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::do3SidedSearchDelegation(const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																PointStructTemplate<T, IDType, num_IDs> *const res_pt_arr_d,
																const T &min_dim1_val,
																const T &max_dim1_val,
																const T &curr_node_med_dim1_val,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char &search_code,
																unsigned char *const &search_codes_arr)
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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(const bool range_split_poss,
																const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																PointStructTemplate<T, IDType, num_IDs> *const res_pt_arr_d,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char &search_code,
																unsigned char *const &search_codes_arr)
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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doRightSearchDelegation(const bool range_split_poss,
																const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																PointStructTemplate<T, IDType, num_IDs> *const res_pt_arr_d,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char &search_code,
																unsigned char *const &search_codes_arr)
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
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAllNodesDelegation(const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																PointStructTemplate<T, IDType, num_IDs> *const res_pt_arr_d,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char *const &search_codes_arr)
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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::splitLeftSearchWork(T *const &root_d,
																const size_t &num_elem_slots,
																const size_t &target_node_ind,
																PointStructTemplate<T, IDType, num_IDs> *const res_pt_arr_d,
																const T &max_dim1_val,
																const T &min_dim2_val,
																long long *const &search_inds_arr,
																unsigned char *const &search_codes_arr)
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
		twoSidedLeftSearchGlobal<<<1, blockDim.x, blockDim.x * (sizeof(long long) + sizeof(unsigned char)), cudaStreamFireAndForget>>>
			(root_d, num_elem_slots, target_node_ind, res_pt_arr_d, max_dim1_val, min_dim2_val);
	}
	else	// Inactive thread has ID i
	{
		search_inds_arr[i] = target_node_ind;
		search_codes_arr[i] = LEFT_SEARCH;
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::splitReportAllNodesWork(T *const &root_d,
																const size_t &num_elem_slots,
																const size_t &target_node_ind,
																PointStructTemplate<T, IDType, num_IDs> *const res_pt_arr_d,
																const T &min_dim2_val,
																long long *const &search_inds_arr,
																unsigned char *const &search_codes_arr)
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
		reportAllNodesGlobal<<<1, blockDim.x, blockDim.x * sizeof(long long), cudaStreamFireAndForget>>>
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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detInactivity(long long &search_ind,
																long long *const &search_inds_arr,
																bool &cont_iter,
																unsigned char *const search_code_ptr,
																unsigned char *const &search_codes_arr)
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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename U, size_t num_U_subarrs, typename V, size_t num_V_subarrs>
__forceinline__ size_t StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::calcTotArrSizeNumUs<U, num_U_subarrs, V, num_V_subarrs>(const size_t num_elem_slots)
	requires SizeOfUAtLeastSizeOfV<U, V>
{
	/*
		tot_arr_size_num_Us = ceil(1/sizeof(U) * num_elem_slots * (sizeof(U) * num_U_subarrs + sizeof(V) * num_V_subarrs + 1 B/bitcode * 1 bitcode))
			With integer truncation:
				if tot_arr_size_bytes % sizeof(U) != 0:
							= tot_arr_size_bytes + 1
				if tot_arr_size_bytes % sizeof(U) == 0:
							= tot_arr_size_bytes
	*/
	// Calculate total size in bytes
	size_t tot_arr_size_bytes = num_elem_slots * (sizeof(U) * num_U_subarrs + sizeof(V) * num_V_subarrs + 1);
	// Divide by sizeof(U)
	size_t tot_arr_size_num_Us = tot_arr_size_bytes / sizeof(U);
	// If tot_arr_size_bytes % sizeof(U) != 0, then tot_arr_size_num_Us * sizeof(U) < tot_arr_size_bytes, so add 1 to tot_arr_size_num_Us
	if (tot_arr_size_bytes % sizeof(U) != 0)
		tot_arr_size_num_Us++;
	return tot_arr_size_num_Us;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __host__ __device__ size_t StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::expOfNextGreaterPowerOf2(const size_t num)
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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __host__ __device__ long long StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::binarySearch(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr, size_t *const &dim1_val_ind_arr, PointStructTemplate<T, IDType, num_IDs> &elem_to_find, const size_t &init_ind, const size_t &num_elems)
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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::printRecur(std::ostream &os, T *const &tree_root, const size_t curr_ind, const size_t num_elem_slots, std::string prefix, std::string child_prefix) const
{
	os << prefix << '(' << getDim1ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< ", " << getDim2ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< "; " << getMedDim1ValsRoot(tree_root, num_elem_slots)[curr_ind];
	if constexpr (num_IDs == 1)
		os << "; " << getIDsRoot(tree_root, num_elem_slots)[curr_ind];
	os << ')';
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
