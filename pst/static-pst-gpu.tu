#include <string>						// To use string-building functions
#include <thrust/execution_policy.h>	// To use thrust::cuda::par::on() stream-specifying execution policy for sorting
#include <thrust/sort.h>				// To use parallel sorting algorithm

#include "arr-ind-assign.h"
#include "err-chk.h"
#include "gpu-tree-node.h"

// C++ allows trailing template type arguments and function parameters to have default values; for template type arguments, it is forbidden for default arguments to be specified for a class template member outside of the class template; for function parameters, one must not declare the default arguments again (as it is regarded as a redefinition, even if the values are the same)
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::StaticPSTGPU(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d,
																	size_t num_elems,
																	const int warp_multiplier,
																	int dev_ind, int num_devs,
																	cudaDeviceProp dev_props)
	// Member initialiser list must be followed by definition
	/*
		Minimum number of array slots necessary to construct any complete tree with num_elems elements is 1 less than the smallest power of 2 greater than num_elems
		Tree is fully balanced by construction, with the placement of nodes in the partially empty last row being deterministic, but not necessarily with the same alignment for a given total number of elements
	*/
	// Number of element slots in each container subarray is nextGreaterPowerOf2(num_elems) - 1
	: num_elem_slots(num_elems == 0 ? 0 : calcNumElemSlots(num_elems)),
	num_elems(num_elems),
	dev_ind(dev_ind),
	num_devs(num_devs),
	dev_props(dev_props)
{
#ifdef DEBUG_CONSTR
	std::cout << "Began constructor\n";
#endif

	if (num_elems == 0)
	{
		root_d = nullptr;
		return;
	}

	const size_t tot_arr_size_num_max_data_id_types = calcTotArrSizeNumMaxDataIDTypes(num_elems);

#ifdef DEBUG_CONSTR
	std::cout << "Ready to allocate memory\n";
#endif

	// Asynchronous memory transfer only permitted for on-host pinned (page-locked) memory, so do such operations in the default stream
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
	{
		// Allocate as a T array so that alignment requirements for larger data types are obeyed
		gpuErrorCheck(cudaMalloc(&root_d, tot_arr_size_num_max_data_id_types * sizeof(T)),
						"Error in allocating priority search tree storage array on device "
						+ std::to_string(dev_ind + 1) + " (1-indexed) of "
						+ std::to_string(num_devs) + ": "
					);
	}
	else
	{
 		if constexpr (sizeof(T) >= sizeof(IDType))
		{
			// Allocate as a T array so that alignment requirements for larger data types are obeyed
			gpuErrorCheck(cudaMalloc(&root_d, tot_arr_size_num_max_data_id_types * sizeof(T)),
							"Error in allocating priority search tree storage array on device "
							+ std::to_string(dev_ind + 1) + " (1-indexed) of "
							+ std::to_string(num_devs) + ": "
						);
		}
		else
		{
			// Allocate as an IDType array so that alignment requirements for larger data types are obeyed
			gpuErrorCheck(cudaMalloc(&root_d, tot_arr_size_num_max_data_id_types * sizeof(IDType)),
							"Error in allocating priority search tree storage array on device "
							+ std::to_string(dev_ind + 1) + " (1-indexed) of "
							+ std::to_string(num_devs) + ": "
						);
		}
	}


	// Create GPU-side array of PointStructTemplate<T, IDType, num_IDs> indices to store sorted results; as this array is not meant to be permanent, avoid storing the two arrays as one contiguous array in order to avoid allocation failure due to global memory fragmentation
	size_t *dim1_val_ind_arr_d;
	size_t *dim2_val_ind_arr_d;
	size_t *dim2_val_ind_arr_secondary_d;


	gpuErrorCheck(cudaMalloc(&dim1_val_ind_arr_d, num_elems * sizeof(size_t)),
					"Error in allocating array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 1 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaMalloc(&dim2_val_ind_arr_d, num_elems * sizeof(size_t)),
					"Error in allocating array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaMalloc(&dim2_val_ind_arr_secondary_d, num_elems * sizeof(size_t)),
					"Error in allocating secondary array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

#ifdef DEBUG_CONSTR
	std::cout << "Allocated index arrays\n";
#endif

	// Synchronous allocations and blocking memory copies complete; do asynchronous initialisations
	// Create asynchronous stream for initialising to 0, as cudaStreamFireAndForget is only available on device
	cudaStream_t stream_root_init;
	gpuErrorCheck(cudaStreamCreateWithFlags(&stream_root_init, cudaStreamNonBlocking),
					"Error in creating asynchronous stream for zero-intialising priority search tree storage array on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
	{
		gpuErrorCheck(cudaMemsetAsync(root_d, 0, tot_arr_size_num_max_data_id_types * sizeof(T),
										stream_root_init),
						"Error in zero-intialising priority search tree storage array via cudaMemset() on device "
						+ std::to_string(dev_ind + 1) + " (1-indexed) of "
						+ std::to_string(num_devs) + ": "
					);
	}
	else
	{
		if constexpr (sizeof(T) >= sizeof(IDType))
		{
#ifdef DEBUG_CONSTR
			std::cout << "About to do an async memory assignment\n";
#endif
			gpuErrorCheck(cudaMemsetAsync(root_d, 0, tot_arr_size_num_max_data_id_types * sizeof(T),
											stream_root_init),
							"Error in zero-intialising priority search tree storage array via cudaMemset() on device "
							+ std::to_string(dev_ind + 1) + " (1-indexed) of "
							+ std::to_string(num_devs) + ": "
						);
		}
		else
		{
			gpuErrorCheck(cudaMemsetAsync(root_d, 0, tot_arr_size_num_max_data_id_types * sizeof(IDType),
											stream_root_init),
							"Error in zero-intialising priority search tree storage array via cudaMemset() on device "
							+ std::to_string(dev_ind + 1) + " (1-indexed) of "
							+ std::to_string(num_devs) + ": "
						);
		}
	}
	// cudaStreamDestroy() is also a kernel submitted to the indicated stream, so it only runs once all previous calls have completed
	gpuErrorCheck(cudaStreamDestroy(stream_root_init),
					"Error in destroying asynchronous stream for zero-intialising priority search tree storage array on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

#ifdef DEBUG_CONSTR
	std::cout << "About to assign index as values to index arrays\n";
#endif

	const size_t index_assign_threads_per_block = warp_multiplier * dev_props.warpSize;
	const size_t index_assign_num_blocks = std::min(num_elems % index_assign_threads_per_block == 0 ?
														num_elems/index_assign_threads_per_block
														: num_elems/index_assign_threads_per_block + 1,
													// static_cast to size_t necessary as dev_props.warpSize is of type int, and std::min fails to compile on arguments of different types
													static_cast<size_t>(dev_props.warpSize * dev_props.warpSize));

#ifdef CONSTR_TIMED
	cudaEvent_t ind1_assign_start, ind1_assign_stop;
	cudaEvent_t ind1_sort_start, ind1_sort_stop;
	cudaEvent_t ind2_assign_start, ind2_assign_stop;
	cudaEvent_t ind2_sort_start, ind2_sort_stop;
	cudaEvent_t ind_proc_sync_start, ind_proc_sync_stop;
	cudaEvent_t populate_tree_start, populate_tree_stop;

	gpuErrorCheck(cudaEventCreate(&ind1_assign_start),
					"Error in creating start event for timing CUDA PST constructor dimension-1 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&ind1_assign_stop),
					"Error in creating stop event for timing CUDA PST constructor dimension-1 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&ind1_sort_start),
					"Error in creating start event for timing CUDA PST constructor dimension-1-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&ind1_sort_stop),
					"Error in creating stop event for timing CUDA PST constructor dimension-1-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	gpuErrorCheck(cudaEventCreate(&ind2_assign_start),
					"Error in creating start event for timing CUDA PST constructor dimension-2 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&ind2_assign_stop),
					"Error in creating stop event for timing CUDA PST constructor dimension-2 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&ind2_sort_start),
					"Error in creating start event for timing CUDA PST constructor dimension-2-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&ind2_sort_stop),
					"Error in creating stop event for timing CUDA PST constructor dimension-2-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	gpuErrorCheck(cudaEventCreate(&ind_proc_sync_start),
					"Error in creating start event for timing synchronisation after CUDA PST constructor index processing on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&ind_proc_sync_stop),
					"Error in creating stop event for timing synchronisation after CUDA PST constructor index processing on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	gpuErrorCheck(cudaEventCreate(&populate_tree_start),
					"Error in creating start event for timing CUDA PST tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&populate_tree_stop),
					"Error in creating stop event for timing CUDA PST tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	gpuErrorCheck(cudaEventRecord(ind_proc_sync_start),
					"Error in recording start event for timing synchronisation after CUDA PST constructor index processing on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	// Create concurrent streams for index-initialising and sorting the dimension-1 and dimension-2 index arrays
	cudaStream_t stream_dim1;
	gpuErrorCheck(cudaStreamCreateWithFlags(&stream_dim1, cudaStreamNonBlocking),
					"Error in creating asynchronous stream for assignment and sorting of indices by dimension 1 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(ind1_assign_start, stream_dim1),
					"Error in recording start event for timing CUDA PST constructor dimension-1 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	arrIndAssign<<<index_assign_num_blocks, index_assign_threads_per_block, 0, stream_dim1>>>(dim1_val_ind_arr_d, num_elems);

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(ind1_assign_stop, stream_dim1),
					"Error in recording stop event for timing CUDA PST constructor dimension-1 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	gpuErrorCheck(cudaEventRecord(ind1_sort_start, stream_dim1),
					"Error in recording start event for timing CUDA PST constructor dimension-1-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	// Sort dimension-1 values index array in ascending order; in-place sort using a curried comparison function; guaranteed O(n) running time or better
	// Execution policy of thrust::cuda::par.on(stream_dim1) guarantees kernel is submitted to stream_dim1
	thrust::sort(thrust::cuda::par.on(stream_dim1), dim1_val_ind_arr_d, dim1_val_ind_arr_d + num_elems,
					Dim1ValIndCompIncOrd(pt_arr_d));

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(ind1_sort_stop, stream_dim1),
					"Error in recording stop event for timing CUDA PST constructor dimension-1-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	// cudaStreamDestroy() is also a kernel submitted to the indicated stream, so it only runs once all previous calls have completed
	gpuErrorCheck(cudaStreamDestroy(stream_dim1),
					"Error in destroying asynchronous stream for assignment and sorting of indices by dimension 1 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);


	cudaStream_t stream_dim2;
	gpuErrorCheck(cudaStreamCreateWithFlags(&stream_dim2, cudaStreamNonBlocking),
					"Error in creating asynchronous stream for assignment and sorting of indices by dimension 2 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(ind2_assign_start, stream_dim2),
					"Error in recording start event for timing CUDA PST constructor dimension-2 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	arrIndAssign<<<index_assign_num_blocks, index_assign_threads_per_block, 0, stream_dim2>>>(dim2_val_ind_arr_d, num_elems);

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(ind2_assign_stop, stream_dim2),
					"Error in recording stop event for timing CUDA PST constructor dimension-2 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	gpuErrorCheck(cudaEventRecord(ind2_sort_start, stream_dim2),
					"Error in recording start event for timing CUDA PST constructor dimension-2-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	// Sort dimension-2 values index array in descending order; in-place sort using a curried comparison function; guaranteed O(n) running time or better
	thrust::sort(thrust::cuda::par.on(stream_dim2), dim2_val_ind_arr_d, dim2_val_ind_arr_d + num_elems,
					Dim2ValIndCompDecOrd(pt_arr_d));

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(ind2_sort_stop, stream_dim2),
					"Error in recording stop event for timing CUDA PST constructor dimension-2-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	gpuErrorCheck(cudaStreamDestroy(stream_dim2),
					"Error in destroying asynchronous stream for assignment and sorting of indices by dimension 2 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	// For correctness, must wait for all streams doing pre-construction pre-processing work to complete before continuing
	gpuErrorCheck(cudaDeviceSynchronize(), "Error in synchronizing with device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs)
					+ " after tree pre-construction pre-processing: "
				);

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(ind_proc_sync_stop),
					"Error in recording stop event for timing synchronisation after CUDA PST constructor index processing on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	gpuErrorCheck(cudaEventRecord(populate_tree_start),
					"Error in recording start event for timing tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	// Populate tree with a one-block grid and a number of threads per block that is a multiple of the warp size
	populateTree<<<1, warp_multiplier * dev_props.warpSize,
					warp_multiplier * dev_props.warpSize * sizeof(size_t) * num_constr_working_arrs>>>
				(root_d, num_elem_slots, pt_arr_d, dim1_val_ind_arr_d, dim2_val_ind_arr_d, dim2_val_ind_arr_secondary_d, 0, num_elems, 0);

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(populate_tree_stop),
					"Error in recording stop event for timing tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	// All threads have finished using these arrays; free them and return
	gpuErrorCheck(cudaFree(dim1_val_ind_arr_d),
					"Error in freeing array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 1 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaFree(dim2_val_ind_arr_d),
					"Error in freeing array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaFree(dim2_val_ind_arr_secondary_d), 
					"Error in freeing secondary array of PointStructTemplate<T, IDType, num_IDs> indices ordered by dimension 2 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventSynchronize(populate_tree_stop),
					"Error in blocking CPU execution until completion of stop event for timing CUDA PST tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);

	float ms = 0;	// milliseconds

	gpuErrorCheck(cudaEventElapsedTime(&ms, ind1_assign_start, ind1_assign_stop),
					"Error in calculating time elapsed for CUDA PST dimension-1 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	std::cout << "CUDA PST dimension-1 index assignment time: " << ms << " ms\n";

	gpuErrorCheck(cudaEventElapsedTime(&ms, ind1_sort_start, ind1_sort_stop),
					"Error in calculating time elapsed for CUDA PST dimension-1-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	std::cout << "CUDA PST dimension-1-based index sorting time: " << ms << " ms\n";

	gpuErrorCheck(cudaEventElapsedTime(&ms, ind2_assign_start, ind2_assign_stop),
					"Error in calculating time elapsed for CUDA PST dimension-2 index assignment on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	std::cout << "CUDA PST dimension-2 index assignment time: " << ms << " ms\n";

	gpuErrorCheck(cudaEventElapsedTime(&ms, ind2_sort_start, ind2_sort_stop),
					"Error in calculating time elapsed for CUDA PST dimension-2-based index sorting on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	std::cout << "CUDA PST dimension-2-based index sorting time: " << ms << " ms\n";

	gpuErrorCheck(cudaEventElapsedTime(&ms, ind_proc_sync_start, ind_proc_sync_stop),
					"Error in calculating overall time elapsed for CUDA PST index processing on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	std::cout << "CUDA PST index processing overall time (from point of view of default stream): " << ms << " ms\n";

	gpuErrorCheck(cudaEventElapsedTime(&ms, populate_tree_start, populate_tree_stop),
					"Error in calculating time elapsed for CUDA PST tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	std::cout << "CUDA PST tree-populating code overall time: " << ms << " ms\n";
#endif
}


// const keyword after method name indicates that the method does not modify any data members of the associated class
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
	if (num_elems == 0)
	{
		os << "Tree is empty\n";
		return;
	}

	const size_t tot_arr_size_num_max_data_id_types = calcTotArrSizeNumMaxDataIDTypes(num_elems);
	T *temp_root;
	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		// No IDs present
		temp_root = new T[tot_arr_size_num_max_data_id_types]();
	else
	{
		if constexpr(sizeof(T) >= sizeof(IDType))
			// sizeof(T) >= sizeof(IDType), so calculate total array size in units of sizeof(T) so that datatype T's alignment requirements will be satisfied
			temp_root = new T[tot_arr_size_num_max_data_id_types]();
		else
			// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
			temp_root = reinterpret_cast<T *>(new IDType[tot_arr_size_num_max_data_id_types]());
	}
	
	if (temp_root == nullptr)
		throwErr("Error: could not allocate " + std::to_string(num_elem_slots)
					+ " elements of type " + typeid(T).name() + "to temp_root");

	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
	{
		gpuErrorCheck(cudaMemcpy(temp_root, root_d, tot_arr_size_num_max_data_id_types * sizeof(T), cudaMemcpyDefault),
						"Error in copying array underlying StaticPSTGPU instance from device to host: ");
	}
	else
	{
		if constexpr (sizeof(T) >= sizeof(IDType))
		{
		gpuErrorCheck(cudaMemcpy(temp_root, root_d, tot_arr_size_num_max_data_id_types * sizeof(T), cudaMemcpyDefault),
						"Error in copying array underlying StaticPSTGPU instance from device to host: ");
		}
		else
		{
			gpuErrorCheck(cudaMemcpy(temp_root, root_d, tot_arr_size_num_max_data_id_types * sizeof(IDType), cudaMemcpyDefault),
							"Error in copying array underlying StaticPSTGPU instance from device to host: ");
		}
	}

	std::string prefix = "";
	std::string child_prefix = "";
	printRecur(os, temp_root, 0, num_elem_slots, prefix, child_prefix);

	delete[] temp_root;
}


// Public search functions

// Default template argument for a class template's member function can only be specified within the class template; similarly, default arguments for functions can only be specified within the class/class template declaration
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::threeSidedSearch(size_t &num_res_elems,
																				RetType *&res_arr_d,
																				T min_dim1_val,
																				T max_dim1_val,
																				T min_dim2_val,
																				const int warp_multiplier
																			)
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
	threeSidedSearchGlobal<T, PointStructTemplate, IDType, num_IDs, RetType>
						  <<<1, warp_multiplier * dev_props.warpSize,
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
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::twoSidedLeftSearch(size_t &num_res_elems,
																				RetType *&res_arr_d,
																				T max_dim1_val,
																				T min_dim2_val,
																				const int warp_multiplier
																			)
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
	// For sufficiently complicated code (such as this one), the compiler cannot deduce types on its own, so supply the (template) types explicitly here
	twoSidedLeftSearchGlobal<T, PointStructTemplate, IDType, num_IDs, RetType>
							<<<1, warp_multiplier * dev_props.warpSize,
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
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::twoSidedRightSearch(size_t &num_res_elems,
																				RetType *&res_arr_d,
																				T min_dim1_val,
																				T min_dim2_val,
																				const int warp_multiplier
																			)
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
	twoSidedRightSearchGlobal<T, PointStructTemplate, IDType, num_IDs, RetType>
							 <<<1, warp_multiplier * dev_props.warpSize,
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
}


// static keyword should only be used when declaring a function in the header file
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
size_t StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::calcGlobalMemNeeded(const size_t num_elems)
{
	const size_t tot_arr_size_num_max_data_id_types = calcTotArrSizeNumMaxDataIDTypes(num_elems);

	size_t global_mem_needed = tot_arr_size_num_max_data_id_types;
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		// No IDs present
		global_mem_needed *= sizeof(T);
	else
	{
		// Separate size-comparison condition from the num_IDs==0 condition so that sizeof(IDType) is well-defined here, as often only one branch of a constexpr if is compiled
		if constexpr (sizeof(T) >= sizeof(IDType))
			global_mem_needed *= sizeof(T);
		else
			global_mem_needed *= sizeof(IDType);
	}

	/*
		Space needed for instantiation = tree size + addend, where addend = max(construction overhead, search overhead) = max(3 * num_elems * size of PointStructTemplate indices, num_elems * size of PointStructTemplate)
		Enough space to contain 3 size_t indices for every node is needed because the splitting of pointers in the dim2_val array at each node creates a need for the dim2_val arrays to be duplicated
		Space needed for reporting nodes is at most num_elems (if all elements are reported) * size of PointStructTemplate (if RetType = PointStructTemplate)
	*/
	const size_t construct_mem_overhead = num_elems * num_constr_working_arrs * sizeof(size_t);
	const size_t search_mem_max_overhead = num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>);
	global_mem_needed += (construct_mem_overhead > search_mem_max_overhead ? construct_mem_overhead : search_mem_max_overhead);

	return global_mem_needed;
}


// Construction-related helper functions

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ long long StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::binarySearch(
																	PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d,
																	size_t *const &dim1_val_ind_arr_d,
																	PointStructTemplate<T, IDType, num_IDs> &elem_to_find,
																	const size_t &init_ind,
																	const size_t &num_elems)
{
	size_t low_ind = init_ind;
	size_t high_ind = init_ind + num_elems;
	size_t mid_ind;		// Avoid reinstantiating mid_ind in every iteration
	// Search is done in the range [low_ind, high_ind)
	while (low_ind < high_ind)
	{
		mid_ind = (low_ind + high_ind)/2;
		// Location in dim1_val_ind_arr_d of elem_to_find has been found
		if (pt_arr_d[dim1_val_ind_arr_d[mid_ind]] == elem_to_find
			&& pt_arr_d[dim1_val_ind_arr_d[mid_ind]].comparisonTiebreaker(elem_to_find) == 0)
			return mid_ind;
		// elem_to_find is before middle element; recurse on left subarray
		else if (elem_to_find.compareDim1(pt_arr_d[dim1_val_ind_arr_d[mid_ind]]) < 0)
			high_ind = mid_ind;
		// elem_to_find is after middle element; recurse on right subarray
		else	// elem_to_find.compareDim1(pt_arr_d[dim1_val_ind_arr_d[mid_ind]]) > 0
			low_ind = mid_ind + 1;
	}
	return -1;	// Element not found
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::constructNode(
																T *const &root_d,
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
		GPUTreeNode::setLeftChild(getBitcodesRoot(root_d, num_elem_slots), target_node_ind);

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
		GPUTreeNode::setRightChild(getBitcodesRoot(root_d, num_elem_slots), target_node_ind);

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
	if (GPUTreeNode::hasChildren(getBitcodesRoot(root_d, num_elem_slots)[target_node_ind]))
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


// Search-related helper functions

// Separate template clauses are necessary when the enclosing template class has different template types from the member function
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::do3SidedSearchDelegation(
																const unsigned char &curr_node_bitcode,
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
																unsigned char *const &search_codes_arr)
{
	// Splitting of query is only possible if the current node has two children and min_dim1_val <= curr_node_med_dim1_val <= max_dim1_val; the equality on max_dim1_val is for the edge case where a median point may be duplicated, with one copy going to the left subtree and the other to the right subtree
	if (min_dim1_val <= curr_node_med_dim1_val
			&& curr_node_med_dim1_val <= max_dim1_val)
	{
		// Query splits over median and node has two children; split into 2 two-sided queries
		if (GPUTreeNode::hasLeftChild(curr_node_bitcode)
				&& GPUTreeNode::hasRightChild(curr_node_bitcode))
		{
			// Delegate work of searching right subtree to another thread and/or block
			splitLeftSearchWork(root_d, num_elem_slots, GPUTreeNode::getRightChild(search_ind),
									res_arr_d, max_dim1_val, min_dim2_val,
									search_inds_arr, search_codes_arr);

			// Prepare to search left subtree with a two-sided right search in the next iteration
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = SearchCodes::RIGHT_SEARCH;
		}
		// No right child, so perform a two-sided right query on the left child
		else if (GPUTreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = SearchCodes::RIGHT_SEARCH;
		}
		// No left child, so perform a two-sided left query on the right child
		else
		{
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getRightChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = SearchCodes::LEFT_SEARCH;
		}
	}
	// Perform three-sided search on left child
	else if (max_dim1_val < curr_node_med_dim1_val
				&& GPUTreeNode::hasLeftChild(curr_node_bitcode))
	{
		// Search code is already a THREE_SEARCH
		search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
	}
	// Perform three-sided search on right child
	// Only remaining possibility, as all others mean the thread is inactive:
	//		curr_node_med_dim1_val < min_dim1_val && GPUTreeNode::hasRightChild(curr_node_bitcode)
	else
	{
		// Search code is already a THREE_SEARCH
		search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getRightChild(search_ind);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doLeftSearchDelegation(
																const bool range_split_poss,
																const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																RetType *const res_arr_d,
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
		if (GPUTreeNode::hasLeftChild(curr_node_bitcode)
				&& GPUTreeNode::hasRightChild(curr_node_bitcode))
		{
			// Delegate work of reporting all nodes in left child to another thread and/or block
			splitReportAllNodesWork(root_d, num_elem_slots, GPUTreeNode::getLeftChild(search_ind),
										res_arr_d, min_dim2_val,
										search_inds_arr, search_codes_arr);


			// Prepare to search right subtree in the next iteration
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getRightChild(search_ind);
		}
		// Node only has a left child; report all on left child
		else if (GPUTreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = REPORT_ALL;
		}
		// Node only has a right child; search on right child
		else if (GPUTreeNode::hasRightChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getRightChild(search_ind);
		}
	}
	// !split_range_poss
	// Only left subtree can possibly contain valid entries; search left subtree
	else if (GPUTreeNode::hasLeftChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doRightSearchDelegation(
																const bool range_split_poss,
																const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																RetType *const res_arr_d,
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
		if (GPUTreeNode::hasLeftChild(curr_node_bitcode)
				&& GPUTreeNode::hasRightChild(curr_node_bitcode))
		{
			// Delegate work of reporting all nodes in right child to another thread and/or block
			splitReportAllNodesWork(root_d, num_elem_slots, GPUTreeNode::getRightChild(search_ind),
										res_arr_d, min_dim2_val,
										search_inds_arr, search_codes_arr);

			// Continue search in the next iteration
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
		}
		// Node only has a right child; report all on right child
		else if (GPUTreeNode::hasRightChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getRightChild(search_ind);
			search_codes_arr[threadIdx.x] = search_code = REPORT_ALL;
		}
		// Node only has a left child; search on left child
		else if (GPUTreeNode::hasLeftChild(curr_node_bitcode))
		{
			search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
		}
	}
	// !range_split_poss
	// Only right subtree can possibly contain valid entries; search right subtree
	else if (GPUTreeNode::hasRightChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getRightChild(search_ind);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::doReportAllNodesDelegation(
																const unsigned char &curr_node_bitcode,
																T *const &root_d,
																const size_t &num_elem_slots,
																RetType *const res_arr_d,
																const T &min_dim2_val,
																long long &search_ind,
																long long *const &search_inds_arr,
																unsigned char *const &search_codes_arr)
{
	if (GPUTreeNode::hasLeftChild(curr_node_bitcode)
			&& GPUTreeNode::hasRightChild(curr_node_bitcode))
	{
		// Delegate reporting of all nodes in right child to another thread and/or block
		splitReportAllNodesWork(root_d, num_elem_slots, GPUTreeNode::getRightChild(search_ind),
									res_arr_d, min_dim2_val,
									search_inds_arr, search_codes_arr);

		// Prepare for next iteration; because execution is already in this branch, search_codes_arr[threadIdx.x] == REPORT_ALL already
		search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
	}
	// Node only has a left child; report all on left child
	else if (GPUTreeNode::hasLeftChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getLeftChild(search_ind);
	}
	// Node only has a right child; report all on right child
	else if (GPUTreeNode::hasRightChild(curr_node_bitcode))
	{
		search_inds_arr[threadIdx.x] = search_ind = GPUTreeNode::getRightChild(search_ind);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::splitLeftSearchWork(
																T *const &root_d,
																const size_t &num_elem_slots,
																const size_t &target_node_ind,
																RetType *const res_arr_d,
																const T &max_dim1_val,
																const T &min_dim2_val,
																long long *const &search_inds_arr,
																unsigned char *const &search_codes_arr)
{
	// Find next inactive thread by iterating through search_inds_arr atomically
	// offset < blockDim.x check comes before atomicCAS() operation because short-circuit evaluation and wrapping of result will ensure atomicCAS() does not activate after all potential target indices have been checked (of which none were empty)
	// atomicCAS(addr, cmp, val) takes the value old := *addr, sets *addr = (old == cmp ? val : old) and returns old; the swap took place iff the return value old == cmp; all calculations are done as one atomic operation
	// Casting necessary to satisfy atomicCAS()'s signature of unsigned long long
	// Target thread ID is (threadIdx.x + offset) % blockDim.x so that no thread starts on the same array element (minimising likelihood of collisions when doing atomic operations), while all potential candidate threads are still checked (self not included, as it will never be inactive)
	size_t offset = 1;
	while (offset < blockDim.x
			&& static_cast<long long>(atomicCAS(reinterpret_cast<unsigned long long *>(search_inds_arr + (threadIdx.x + offset) % blockDim.x),
												static_cast<unsigned long long>(INACTIVE_IND),
												target_node_ind))
										!= INACTIVE_IND)
	{offset++;}
	// Upon exit, offset either contains the index of the thread that will report all nodes in the corresponding subtree; or offset >= blockDim.x
	if (offset >= blockDim.x)	// No inactive threads; use dynamic parallelism
	{
		// report-all searches never become normal searches again, so do not need shared memory for a search_codes_arr, just a search_inds_arr
		// For sufficiently complicated code (such as this one), the compiler cannot deduce types on its own, so supply the (template) types explicitly here
		// Note that cudaStreamFireAndForget is not defined with the legacy CUDA debugger backend, and so cannot be used with such debuggers
		twoSidedLeftSearchGlobal<T, PointStructTemplate, IDType, num_IDs, RetType>
								<<<1, blockDim.x, blockDim.x * (sizeof(long long) + sizeof(unsigned char)),
									cudaStreamFireAndForget>>>
			(root_d, num_elem_slots, target_node_ind, res_arr_d, max_dim1_val, min_dim2_val);
	}
	else	// Inactive thread has ID (threadIdx.x + offset) % blockDim.x
	{
		search_inds_arr[(threadIdx.x + offset) % blockDim.x] = target_node_ind;
		search_codes_arr[(threadIdx.x + offset) % blockDim.x] = LEFT_SEARCH;
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
		>::value
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::splitReportAllNodesWork(
																T *const &root_d,
																const size_t &num_elem_slots,
																const size_t &target_node_ind,
																RetType *const res_arr_d,
																const T &min_dim2_val,
																long long *const &search_inds_arr,
																unsigned char *const &search_codes_arr)
{
	// Find next inactive thread by iterating through search_inds_arr atomically
	// offset < blockDim.x check comes before atomicCAS() operation because short-circuit evaluation and wrapping of result will ensure atomicCAS() does not activate after all potential target indices have been checked (of which none were empty)
	// atomicCAS(addr, cmp, val) takes the value old := *addr, sets *addr = (old == cmp ? val : old) and returns old; the swap took place iff the return value old == cmp; all calculations are done as one atomic operation
	// Casting necessary to satisfy atomicCAS()'s signature of unsigned long long
	// Target thread ID is (threadIdx.x + offset) % blockDim.x so that no thread starts on the same array element (minimising likelihood of collisions when doing atomic operations), while all potential candidate threads are still checked (self not included, as it will never be inactive)
	size_t offset = 1;
	while (offset < blockDim.x
			&& static_cast<long long>(atomicCAS(reinterpret_cast<unsigned long long *>(search_inds_arr + (threadIdx.x + offset) % blockDim.x),
												static_cast<unsigned long long>(INACTIVE_IND),
												target_node_ind))
										!= INACTIVE_IND)
	{offset++;}
	// Upon exit, offset either contains the index of the thread that will report all nodes in the corresponding subtree; or offset >= blockDim.x
	if (offset >= blockDim.x)	// No inactive threads; use dynamic parallelism
	{
		// report-all searches never become normal searches again, so do not need shared memory for a search_codes_arr, just a search_inds_arr
		// For sufficiently complicated code (such as this one), the compiler cannot deduce types on its own, so supply the (template) types explicitly here
		// Note that cudaStreamFireAndForget is not defined with the legacy CUDA debugger backend, and so cannot be used with such debuggers
		reportAllNodesGlobal<T, PointStructTemplate, IDType, num_IDs, RetType>
							<<<1, blockDim.x, blockDim.x * sizeof(long long),
								cudaStreamFireAndForget>>>
			(root_d, num_elem_slots, target_node_ind, res_arr_d, min_dim2_val);
	}
	else	// Inactive thread has ID (threadIdx.x + offset) % blockDim.x
	{
		search_inds_arr[(threadIdx.x + offset) % blockDim.x] = target_node_ind;

		// For splitting of work when not called from reportAllNodesGlobal()
		if (search_codes_arr != nullptr)
			search_codes_arr[(threadIdx.x + offset) % blockDim.x] = REPORT_ALL;
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::detInactivity(
																long long &search_ind,
																long long *const &search_inds_arr,
																bool &cont_iter,
																unsigned char *const search_code_ptr,
																unsigned char *const &search_codes_arr)
{
	// INACTIVE threads check whether they should be active in the next iteration; if not, and all threads are inactive, set iteration toggle to false

	// Thread has been assigned work; update local variables accordingly
	if (search_ind == INACTIVE_IND
			&& search_inds_arr[threadIdx.x] != INACTIVE_IND)
	{
		search_ind = search_inds_arr[threadIdx.x];
		// These two should always have or not have values of nullptr at the same time, but check both just in case
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


// Data footprint calculation functions

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
inline size_t StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::calcTotArrSizeNumMaxDataIDTypes(const size_t num_elems)
{
	// Number of element slots in each container subarray is nextGreaterPowerOf2(num_elems) - 1
	// Class member num_elem_slots is not available due to being in a static function, so must re-calculate it here
	const size_t num_elem_slots = calcNumElemSlots(num_elems);

	// constexpr if is a C++17 feature that only compiles the branch of code that evaluates to true at compile-time, saving executable space and execution runtime
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		// No IDs present
		return calcTotArrSizeNumTs<num_val_subarrs>(num_elem_slots);
	else
	{
		// Separate size-comparison condition from the num_IDs==0 condition so that sizeof(IDType) is well-defined here, as often only one branch of a constexpr if is compiled
		if constexpr (sizeof(T) >= sizeof(IDType))
			// sizeof(T) >= sizeof(IDType), so calculate total array size in units of sizeof(T) so that datatype T's alignment requirements will be satisfied
			return calcTotArrSizeNumUs<T, num_val_subarrs, IDType, num_IDs>(num_elem_slots);
		else
			// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
			return calcTotArrSizeNumUs<IDType, num_IDs, T, num_val_subarrs>(num_elem_slots);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <size_t num_T_subarrs>
inline size_t StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::calcTotArrSizeNumTs(const size_t num_elem_slots)
{
	/*
		tot_arr_size_num_Ts = ceil(1/sizeof(T) * num_elem_slots * (sizeof(T) * num_T_subarrs + 1 B/bitcode * 1 bitcode))
			With integer truncation:
				if tot_arr_size_bytes % sizeof(T) != 0:
							= tot_arr_size_bytes + 1
				if tot_arr_size_bytes % sizeof(T) == 0:
							= tot_arr_size_bytes
	*/
	// Calculate total size in bytes
	size_t tot_arr_size_bytes = num_elem_slots * (sizeof(T) * num_T_subarrs + 1);
	// Divide by sizeof(T)
	size_t tot_arr_size_num_Ts = tot_arr_size_bytes / sizeof(T);
	// If tot_arr_size_bytes % sizeof(T) != 0, then tot_arr_size_num_Ts * sizeof(T) < tot_arr_size_bytes, so add 1 to tot_arr_size_num_Ts
	if (tot_arr_size_bytes % sizeof(T) != 0)
		tot_arr_size_num_Ts++;
	return tot_arr_size_num_Ts;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename U, size_t num_U_subarrs, typename V, size_t num_V_subarrs>
	requires SizeOfUAtLeastSizeOfV<U, V>
inline size_t StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::calcTotArrSizeNumUs<U, num_U_subarrs, V, num_V_subarrs>(const size_t num_elem_slots)
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
void StaticPSTGPU<T, PointStructTemplate, IDType, num_IDs>::printRecur(std::ostream &os,
																		T *const &tree_root,
																		const size_t curr_ind,
																		const size_t num_elem_slots,
																		std::string prefix,
																		std::string child_prefix
																	) const
{
	os << prefix << '(' << getDim1ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< ", " << getDim2ValsRoot(tree_root, num_elem_slots)[curr_ind]
				<< "; " << getMedDim1ValsRoot(tree_root, num_elem_slots)[curr_ind];
	if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		os << "; " << getIDsRoot(tree_root, num_elem_slots)[curr_ind];
	os << ')';
	const unsigned char curr_node_bitcode = getBitcodesRoot(tree_root, num_elem_slots)[curr_ind];
	if (GPUTreeNode::hasLeftChild(curr_node_bitcode)
			&& GPUTreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, GPUTreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "├─(R)─ ", child_prefix + "│      ");
		printRecur(os, tree_root, GPUTreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
	else if (GPUTreeNode::hasRightChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, GPUTreeNode::getRightChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(R)─ ", child_prefix + "       ");
	}
	else if (GPUTreeNode::hasLeftChild(curr_node_bitcode))
	{
		printRecur(os, tree_root, GPUTreeNode::getLeftChild(curr_ind), num_elem_slots,
					'\n' + child_prefix + "└─(L)─ ", child_prefix + "       ");
	}
}
