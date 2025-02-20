#include <algorithm>					// To use std::max()
#include <string>						// To use string-building functions
#include <thrust/execution_policy.h>	// To use thrust::cuda::par::on() stream-specifying execution policy for sorting
#include <thrust/sort.h>				// To use parallel sorting algorithm

#include "arr-ind-assign.h"
#include "class-member-checkers.h"
#include "dev-symbols.h"			// For global memory-scoped variable res_arr_ind_d

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::StaticPSTGPUArr(PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
																			size_t num_elems,
																			const unsigned threads_per_block,
																			int dev_ind, int num_devs,
																			cudaDeviceProp dev_props)
	/*
		All trees except potentially the last tree in the array are complete trees in order to reduce internal fragmentation
		In order to reduce dynamic parallelism cost in construction and communication overhead in search, make each complete tree have enough elements such that each thread is active at least once (so that differing block sizes that are not powers of 2 will have an effect on performance) and will only process at most two elements in the last level (which is the only level where it is possible to have an insufficient number of threads available), allowing for a constant number of resources to handle this (relatively common) edge case
	*/
	: full_tree_num_elem_slots(num_elems == 0 ? 0 : calcNumElemSlotsPerTree(threads_per_block)),
	full_tree_size_num_max_data_id_types(num_elems == 0 ? 0 : calcTreeSizeNumMaxDataIDTypes(full_tree_num_elem_slots)),
	num_elems(num_elems),
	// Each tree (except potentially the last one) contains as many elements as it can hold in order to reduce internal fragmentation and maximise thread occupancy
	// Total number of subtrees = num_thread_blocks = ceil(num_elems/threads_per_block)
	// As full_tree_num_elem_slots is declared before num_thread_blocks in the class declaration, it is also instantiated first
	// When this order is violated, no compilation error is reported; the data member that depends on a later-declared data member is simply incorrectly initialised
	num_thread_blocks(num_elems / full_tree_num_elem_slots + (num_elems % full_tree_num_elem_slots == 0 ? 0 : 1)),
	threads_per_block(threads_per_block),
	dev_ind(dev_ind),
	num_devs(num_devs),
	dev_props(dev_props)
{
#ifdef DEBUG_CONSTR
	std::cout << "Began constructor\n";
#endif

	if (num_elems == 0)
	{
		tree_arr_d = nullptr;
		return;
	}

	const size_t tree_arr_size_num_max_data_id_types = calcTreeArrSizeNumMaxDataIDTypes(num_elems, threads_per_block);

#ifdef DEBUG_CONSTR
	std::cout << "Ready to allocate memory\n";
#endif

	// Asynchronous memory transfer only permitted for on-host pinned (page-locked) memory, so do such operations in the default stream
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
	{
		// Allocate as a T array so that alignment requirements for larger data types are obeyed
		gpuErrorCheck(cudaMalloc(&tree_arr_d, tree_arr_size_num_max_data_id_types * sizeof(T)),
						"Error in allocating array of PSTs on device "
						+ std::to_string(dev_ind + 1) + " (1-indexed) of "
						+ std::to_string(num_devs) + " :"
					);
	}
	else
	{
		// Allocate as an IDType array so that alignment requirements for larger data types are obeyed
		gpuErrorCheck(cudaMalloc(&tree_arr_d, tree_arr_size_num_max_data_id_types * sizeof(IDType)),
						"Error in allocating array of PSTs on device "
						+ std::to_string(dev_ind + 1) + " (1-indexed) of "
						+ std::to_string(num_devs) + " :"
					);
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

#ifdef DEBUG_CONSTR
	std::cout << "About to do an async memory assignment\n";
#endif

	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
	{
		gpuErrorCheck(cudaMemsetAsync(tree_arr_d, 0, tree_arr_size_num_max_data_id_types * sizeof(T),
										stream_root_init),
						"Error in zero-intialising priority search tree storage array via cudaMemset() on device "
						+ std::to_string(dev_ind + 1) + " (1-indexed) of "
						+ std::to_string(num_devs) + ": "
					);
	}
	else
	{
		gpuErrorCheck(cudaMemsetAsync(tree_arr_d, 0, tree_arr_size_num_max_data_id_types * sizeof(IDType),
										stream_root_init),
						"Error in zero-intialising priority search tree storage array via cudaMemset() on device "
						+ std::to_string(dev_ind + 1) + " (1-indexed) of "
						+ std::to_string(num_devs) + ": "
					);
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

	const size_t index_assign_num_blocks = std::min(num_elems % threads_per_block == 0 ?
														num_elems/threads_per_block
														: num_elems/threads_per_block + 1,
													// static_cast to size_t necessary as dev_props.warpSize is of type int, and std::min fails to compile on arguments of different types
													static_cast<size_t>(dev_props.warpSize * dev_props.warpSize));

#ifdef CONSTR_TIMED
	cudaEvent_t ind1_assign_start, ind1_assign_stop;
	cudaEvent_t ind1_sort_start, ind1_sort_stop;
	cudaEvent_t ind2_assign_start, ind2_assign_stop;
	cudaEvent_t ind2_sort_start, ind2_sort_stop;
	cudaEvent_t ind_proc_sync_start, ind_proc_sync_stop;
	cudaEvent_t populate_trees_start, populate_trees_stop;

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

	gpuErrorCheck(cudaEventCreate(&populate_trees_start),
					"Error in creating start event for timing CUDA PST tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
	gpuErrorCheck(cudaEventCreate(&populate_trees_stop),
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

	arrIndAssign<<<index_assign_num_blocks, threads_per_block, 0, stream_dim1>>>(dim1_val_ind_arr_d, num_elems);

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

	// TODO: for now, run sort using multiple kernel calls (as this is easier to do) in order to ascertain whether it is worthwhile to invest in writing a better, one-kernel partitioned sort function (i.e. a sort function that sorts elements within partitions of a given size)

	for (auto i = 0; i < num_thread_blocks - 1; i++)
	{
		// Sort dimension-1 values index array in ascending order; in-place sort using a curried comparison function; guaranteed O(n) running time or better
		// Execution policy of thrust::cuda::par.on(stream_dim1) guarantees kernel is submitted to stream_dim1
		thrust::sort(thrust::cuda::par.on(stream_dim1),
						dim1_val_ind_arr_d + full_tree_num_elem_slots * i,
						dim1_val_ind_arr_d + full_tree_num_elem_slots * (i + 1),
						Dim1ValIndCompIncOrd(pt_arr_d)
					);
	}

	// Last block may be differently sized, so simply make a slightly different sort call for it
	thrust::sort(thrust::cuda::par.on(stream_dim1),
					dim1_val_ind_arr_d + full_tree_num_elem_slots * (num_thread_blocks - 1),
					dim1_val_ind_arr_d + num_elems,
					Dim1ValIndCompIncOrd(pt_arr_d)
				);

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

	arrIndAssign<<<index_assign_num_blocks, threads_per_block, 0, stream_dim2>>>(dim2_val_ind_arr_d, num_elems);

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

	// TODO: for now, run sort using multiple kernel calls (as this is easier to do) in order to ascertain whether it is worthwhile to invest in writing a better, one-kernel partitioned sort function (i.e. a sort function that sorts elements within partitions of a given size)

	for (auto i = 0; i < num_thread_blocks - 1; i++)
	{
		// Sort dimension-2 values index array in descending order; in-place sort using a curried comparison function; guaranteed O(n) running time or better
		// Execution policy of thrust::cuda::par.on(stream_dim2) guarantees kernel is submitted to stream_dim2
		thrust::sort(thrust::cuda::par.on(stream_dim2),
						dim2_val_ind_arr_d + full_tree_num_elem_slots * i,
						dim2_val_ind_arr_d + full_tree_num_elem_slots * (i + 1),
						Dim2ValIndCompDecOrd(pt_arr_d)
					);
	}

	// Last block may be differently sized, so simply make a slightly different sort call for it
	thrust::sort(thrust::cuda::par.on(stream_dim2),
					dim2_val_ind_arr_d + full_tree_num_elem_slots * (num_thread_blocks - 1),
					dim2_val_ind_arr_d + num_elems,
					Dim2ValIndCompDecOrd(pt_arr_d)
				);

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

	gpuErrorCheck(cudaEventRecord(populate_trees_start),
					"Error in recording start event for timing tree-populating code on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of "
					+ std::to_string(num_devs) + ": "
				);
#endif

	// Populate trees with a multi-block grid, with one block per tree
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
	{
		// No ID or sizeof(T) >= sizeof(IDType); full_tree_size_num_max_data_id_types is already in units of sizeof(T)
		populateTrees<<<num_thread_blocks, threads_per_block,
							threads_per_block * sizeof(size_t) * num_constr_working_arrs>>>
						(tree_arr_d, full_tree_num_elem_slots,
						 full_tree_size_num_max_data_id_types,
						 num_elems, pt_arr_d, dim1_val_ind_arr_d,
						 dim2_val_ind_arr_d, dim2_val_ind_arr_secondary_d);
	}
	else
	{
		// sizeof(IDType) > sizeof(T), and the latter is guaranteed to be a factor of the former
		populateTrees<<<num_thread_blocks, threads_per_block,
							threads_per_block * sizeof(size_t) * num_constr_working_arrs>>>
						(tree_arr_d, full_tree_num_elem_slots,
						 full_tree_size_num_max_data_id_types * sizeof(IDType) / sizeof(T),
						 num_elems, pt_arr_d, dim1_val_ind_arr_d,
						 dim2_val_ind_arr_d, dim2_val_ind_arr_secondary_d);
	}

#ifdef CONSTR_TIMED
	gpuErrorCheck(cudaEventRecord(populate_trees_stop),
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
	gpuErrorCheck(cudaEventSynchronize(populate_trees_stop),
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

	gpuErrorCheck(cudaEventElapsedTime(&ms, populate_trees_start, populate_trees_stop),
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
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
	if (num_elems == 0)
	{
		os << "Tree array is empty\n";
		return;
	}

	const size_t tree_arr_size_num_max_data_id_types = calcTreeArrSizeNumMaxDataIDTypes(num_elems, threads_per_block);
	T *temp_tree_arr;
	// Use of () after new and new[] causes value-initialisation (to 0) starting in C++03; needed for any nodes that technically contain no data
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
		// No IDs present or sizeof(T) >= sizeof(IDType), so calculate total array size in units of sizeof(T) so that datatype T's alignment requirements will be satisfied
		temp_tree_arr = new T[tree_arr_size_num_max_data_id_types]();
	else
		// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
		temp_tree_arr = reinterpret_cast<T *>(new IDType[tree_arr_size_num_max_data_id_types]());

	if (temp_tree_arr == nullptr)
	{
		if (num_elems % full_tree_num_elem_slots == 0)		// All trees are full trees
			throwErr("Error: could not allocate memory for tree array containing "
						+ std::to_string(num_thread_blocks) + " trees with "
						+ std::to_string(full_tree_num_elem_slots) + " node slots"
					);
		else
			throwErr("Error: could not allocate memory for tree array containing "
						+ std::to_string(num_thread_blocks) + " trees with "
						+ std::to_string(full_tree_num_elem_slots)
						+ " node slots and 1 final tree with "
						+ std::to_string(num_elems % full_tree_num_elem_slots) + " node slots"
					);
	}

	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
	{
		gpuErrorCheck(cudaMemcpy(temp_tree_arr, tree_arr_d,
									tree_arr_size_num_max_data_id_types * sizeof(T), cudaMemcpyDefault),
						"Error in copying array underlying StaticPSTGPU instance from device to host: ");
	}
	else
	{
		gpuErrorCheck(cudaMemcpy(temp_tree_arr, tree_arr_d,
									tree_arr_size_num_max_data_id_types * sizeof(IDType), cudaMemcpyDefault),
						"Error in copying array underlying StaticPSTGPU instance from device to host: ");
	}

	for (auto i = 0; i < num_thread_blocks; i++)
	{
		std::cout << "Tree " << i + 1 << " (1-indexed) of " << num_thread_blocks << ":\n";
		std::string prefix = "";
		std::string child_prefix = "";

		// Decrease repetitive calculations by saving result
		const bool all_trees_full = num_elems % full_tree_num_elem_slots == 0;

		// Distinguish between number of element slots in final tree and all other trees if necessary
		if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
						|| SizeOfUAtLeastSizeOfV<T, IDType>)
		{
			printRecur(os, temp_tree_arr + full_tree_size_num_max_data_id_types, 0,
						all_trees_full ? full_tree_num_elem_slots
							: (i < num_thread_blocks - 1 ? full_tree_num_elem_slots
								: calcNumElemSlotsPerTree(num_elems % full_tree_num_elem_slots)),
						prefix, child_prefix);
		}
		else
		{
			printRecur(os, temp_tree_arr + full_tree_size_num_max_data_id_types * sizeof(IDType) / sizeof(T),
						0,
						all_trees_full ? full_tree_num_elem_slots
							: (i < num_thread_blocks - 1 ? full_tree_num_elem_slots
								: calcNumElemSlotsPerTree(num_elems % full_tree_num_elem_slots)),
						prefix, child_prefix);
		}
		std::cout << "\n\n";
	}

	delete[] temp_tree_arr;
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
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::threeSidedSearch(size_t &num_res_elems,
																				RetType *&res_arr_d,
																				T min_dim1_val,
																				T max_dim1_val,
																				T min_dim2_val
																			)
{
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::twoSidedLeftSearch(size_t &num_res_elems,
																					RetType *&res_arr_d,
																					T max_dim1_val,
																					T min_dim2_val
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
					+ std::to_string(dev_ind + 1) + " (1-indexed) of " + std::to_string(num_devs)
					+ ": ");

	// Set on-device global result array index to 0
	const unsigned long long res_arr_ind_init = 0;
	// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
	gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind_init, sizeof(size_t),
										0, cudaMemcpyDefault),
					"Error in initialising global result array index to 0 on device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of " + std::to_string(num_devs)
					+ ": ");

	// Call global function for on-device search
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
	{
		// No ID or sizeof(T) >= sizeof(IDType); full_tree_size_num_max_data_id_types is already in units of sizeof(T)
		twoSidedLeftSearchTreeArrGlobal<T, PointStructTemplate, IDType, num_IDs, RetType>
										<<<num_thread_blocks, threads_per_block,
											threads_per_block * (sizeof(long long) + sizeof(unsigned char))>>>
										(tree_arr_d, full_tree_num_elem_slots,
										 full_tree_size_num_max_data_id_types,
										 num_elems, res_arr_d, max_dim1_val, min_dim2_val);
	}
	else
	{
		// sizeof(IDType) > sizeof(T), and the latter is guaranteed to be a factor of the former
		twoSidedLeftSearchTreeArrGlobal<T, PointStructTemplate, IDType, num_IDs, RetType>
										<<<num_thread_blocks, threads_per_block,
											threads_per_block * (sizeof(long long) + sizeof(unsigned char))>>>
										(tree_arr_d, full_tree_num_elem_slots,
										 full_tree_size_num_max_data_id_types * sizeof(IDType) / sizeof(T),
										 num_elems, res_arr_d, max_dim1_val, min_dim2_val);
	}

	// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
	// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
	gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d,
										sizeof(unsigned long long), 0,
										cudaMemcpyDefault),
					"Error in copying global result array final index from device "
					+ std::to_string(dev_ind + 1) + " (1-indexed) of " + std::to_string(num_devs)
					+ ": ");
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename RetType>
			requires std::disjunction<
								std::is_same<RetType, IDType>,
								std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
				>::value
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::twoSidedRightSearch(size_t &num_res_elems,
																					RetType *&res_arr_d,
																					T min_dim1_val,
																					T min_dim2_val
																				)
{
}


// static keyword should only be used when declaring a function in the header file
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcGlobalMemNeeded(const size_t num_elems, const unsigned threads_per_block)
{
	const size_t tree_arr_size_num_max_data_id_types = calcTreeArrSizeNumMaxDataIDTypes(num_elems, threads_per_block);

	size_t global_mem_needed = tree_arr_size_num_max_data_id_types;
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
		// No IDs present or sizeof(T) >= sizeof(IDType)
		global_mem_needed *= sizeof(T);
	else
		global_mem_needed *= sizeof(IDType);

	/*
		Space needed for instantiation = tree array size + addend, where addend = max(construction overhead, search overhead) = max(3 * num_elems * size of PointStructTemplate indices, num_elems * size of PointStructTemplate)
		Enough space to contain 3 size_t indices for every node is needed because the splitting of pointers in the dim2_val array at each node creates a need for the dim2_val arrays to be duplicated
		Space needed for reporting nodes is at most num_elems (if all elements are reported) * size of PointStructTemplate (if RetType = PointStructTemplate)
	*/
	const size_t construct_mem_overhead = num_elems * num_constr_working_arrs * sizeof(size_t);
	const size_t search_mem_max_overhead = num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>);
	global_mem_needed += std::max(construct_mem_overhead, search_mem_max_overhead);

	return global_mem_needed;
}


// Construction-related helper functions

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ long long StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::binarySearch(
															PointStructTemplate<T, IDType, num_IDs> *const pt_arr_d,
															size_t *const dim1_val_ind_arr_d,
															PointStructTemplate<T, IDType, num_IDs> const &elem_to_find,
															const size_t init_ind,
															const size_t num_elems
														)
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
__forceinline__ __device__ void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::constructNode(
																T *const tree_root_d,
																const size_t tree_num_elem_slots,
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
																size_t &right_subarr_num_elems
															)
{
	size_t median_dim1_val_ind;

	// Treat dim1_val_ind_arr_ind[max_dim2_val_dim1_array_ind] as a removed element (but don't actually remove the element for performance reasons

	if (num_subelems == 1)		// Base case
	{
		median_dim1_val_ind = subelems_start_ind;
		left_subarr_num_elems = 0;
		right_subarr_num_elems = 0;
	}
	// max_dim2_val originally comes from the part of the array to the left of median_dim1_val
	else if (max_dim2_val_dim1_array_ind < subelems_start_ind + num_subelems/2)
	{
		// As median values are always put in the left subtree, when the subroot value comes from the left subarray, the median index is given by num_elems/2, which evenly splits the array if there are an even number of elements remaining or makes the left subtree larger by one element if there are an odd number of elements remaining
		median_dim1_val_ind = subelems_start_ind + num_subelems/2;
		// max_dim2_val has been removed from the left subarray, so there are median_dim1_val_ind elements remaining on the left side
		left_subarr_num_elems = median_dim1_val_ind - subelems_start_ind;
		right_subarr_num_elems = subelems_start_ind + num_subelems - median_dim1_val_ind - 1;
	}
	/*
		max_dim2_val originally comes from the part of the array to the right of median_dim1_val, i.e.
			max_dim2_val_dim1_array_ind >= subelems_start_ind + num_subelems/2
	*/
	else
	{
		median_dim1_val_ind = subelems_start_ind + num_subelems/2 - 1;

		left_subarr_num_elems = median_dim1_val_ind - subelems_start_ind + 1;
		right_subarr_num_elems = subelems_start_ind + num_subelems - median_dim1_val_ind - 2;
	}

	setNode(tree_root_d, target_node_ind, tree_num_elem_slots,
			pt_arr_d[dim2_val_ind_arr_d[subelems_start_ind]],
			pt_arr_d[dim1_val_ind_arr_d[median_dim1_val_ind]].dim1_val);

	if (left_subarr_num_elems > 0)
	{
		GPUTreeNode::setLeftChild(getBitcodesRoot(tree_root_d, tree_num_elem_slots), target_node_ind);

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
		GPUTreeNode::setRightChild(getBitcodesRoot(tree_root_d, tree_num_elem_slots), target_node_ind);

		// max_dim2_val is to the right of median_dim1_val_ind in dim1_val_ind_arr_d; shift all entries after max_dim2_val_dim1_array_ind leftward, overwriting max_dim2_val_array_ind
		if (max_dim2_val_dim1_array_ind > median_dim1_val_ind)
			for (size_t i = max_dim2_val_dim1_array_ind;
					i < subelems_start_ind + num_subelems - 1; i++)
				dim1_val_ind_arr_d[i] = dim1_val_ind_arr_d[i+1];
		// Otherwise, max_dim2_val is to the left of median_dim1_val in dim1_val_ind_arr_d; leave right subarray as is
	}


	// Choose median_dim1_val_ind + 1 as the starting index for all right subarrays, as this is the only index that is valid no matter whether max_dim2_val is to the left or right of med_dim1_val
	right_subarr_start_ind = median_dim1_val_ind + 1;


	// Iterate through dim2_val_ind_arr_d, placing data with lower dimension-1 value in the subarray for the left subtree and data with higher dimension-1 value in the subarray for the right subtree
	// Note that because subarrays' sizes were allocated based on this same data, there should not be a need to check that left_dim2_subarr_iter_ind < left_subarr_num_elems (and similarly for the right side)
	if (GPUTreeNode::hasChildren(getBitcodesRoot(tree_root_d, tree_num_elem_slots)[target_node_ind]))
	{
		size_t left_dim2_subarr_iter_ind = subelems_start_ind;
		size_t right_dim2_subarr_iter_ind = right_subarr_start_ind;

		// Skip over first (largest) element in dim2_val_ind_arr_d, as it has been already placed in the current node
		for (size_t i = subelems_start_ind + 1; i < subelems_start_ind + num_subelems; i++)
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


// Data footprint calculation functions

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
inline size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeArrSizeNumMaxDataIDTypes(const size_t num_elems, const unsigned threads_per_block)
{
	// Full trees are trees that are complete and have nextGreaterPowerOf2(threads_per_block) - 1 elements
	const size_t full_tree_num_elem_slots = calcNumElemSlotsPerTree(threads_per_block);
	const size_t full_tree_size_num_max_data_id_types = calcTreeSizeNumMaxDataIDTypes(full_tree_num_elem_slots);

	// Equal to number of thread blocks
	const unsigned num_trees = num_elems / full_tree_num_elem_slots
								+ (num_elems % full_tree_num_elem_slots == 0 ? 0 : 1);

	size_t tree_arr_size_num_max_data_id_types = full_tree_size_num_max_data_id_types;

	// Each tree is a complete tree
	if (num_elems % full_tree_num_elem_slots == 0)
		tree_arr_size_num_max_data_id_types *= num_trees;
	// All trees except for the last are full trees; determine the number of slots necessary for the last tree separately
	else
	{
		tree_arr_size_num_max_data_id_types *= num_trees - 1;

		// Last tree size calculation in units of the larger of data or ID type
		const size_t last_tree_num_elem_slots = calcNumElemSlotsPerTree(num_elems % full_tree_num_elem_slots);
		const size_t last_tree_size_num_max_data_id_types = calcTreeSizeNumMaxDataIDTypes(last_tree_num_elem_slots);

		tree_arr_size_num_max_data_id_types += last_tree_size_num_max_data_id_types;
	}

	return tree_arr_size_num_max_data_id_types;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
inline size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeSizeNumMaxDataIDTypes(const size_t num_elem_slots_per_tree)
{
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value
					|| SizeOfUAtLeastSizeOfV<T, IDType>)
		// No IDs present or sizeof(T) >= sizeof(IDType)
		return calcTreeSizeNumTs<num_val_subarrs>(num_elem_slots_per_tree);
	else
		// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
		return calcTreeSizeNumIDTypes<num_val_subarrs>(num_elem_slots_per_tree);
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <size_t num_T_subarrs>
inline size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeSizeNumTs(const size_t num_elem_slots_per_tree)
{
	/*
		sizeof(T) >= sizeof(IDType), so alignment requirements for all types satisfied when using maximal compaction

		tot_arr_size_num_Ts = ceil(1/sizeof(T) * num_elem_slots * (sizeof(T) * num_T_subarrs + sizeof(IDType) * num_IDs + 1 B/bitcode * 1 bitcode))
	*/
	// Calculate total size in bytes
	size_t tree_size_bytes = sizeof(T) * num_T_subarrs + 1;
	if constexpr (HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		tree_size_bytes += sizeof(IDType) * num_IDs;
	tree_size_bytes *= num_elem_slots_per_tree;

	// Divide by sizeof(T)
	size_t tree_size_num_Ts = tree_size_bytes / sizeof(T);
	// If tree_size_bytes % sizeof(T) != 0, then tree_size_num_Ts * sizeof(T) < tree_size_bytes, so add 1 to tree_size_num_Ts
	if (tree_size_bytes % sizeof(T) != 0)
		tree_size_num_Ts++;
	return tree_size_num_Ts;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <size_t num_T_subarrs>
	requires NonVoidType<IDType>
inline size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeSizeNumIDTypes(const size_t num_elem_slots)
{
	/*
		sizeof(IDType) > sizeof(T), so extra padding must be placed before IDType array to ensure alignment requirements are met (hence the distribution of the ceil() function around each addend

		tot_arr_size_num_IDTypes = ceil(1/sizeof(IDType) * num_elem_slots * sizeof(T) * num_T_subarrs)
									+ num_elem_slots * num_IDs
									+ ceil(1/sizeof(IDType) * num_elem_slots * 1 B/bitcode * 1 bitcode)
	*/
	// Calculate size of value arrays in units of number of IDTypes
	const size_t val_arr_size_bytes = num_elem_slots * sizeof(T) * num_T_subarrs;
	const size_t val_arr_size_num_IDTypes = val_arr_size_bytes / sizeof(IDType)
												+ (val_arr_size_bytes % sizeof(IDType) == 0 ? 0 : 1);

	// Calculate size of bitcode array in units of number of IDTypes
	const size_t bitcode_arr_size_bytes = num_elem_slots;
	const size_t bitcode_arr_size_num_IDTypes = bitcode_arr_size_bytes / sizeof(IDType)
												+ (bitcode_arr_size_bytes  % sizeof(IDType) == 0 ? 0 : 1);

	const size_t tot_arr_size_num_IDTypes = val_arr_size_num_IDTypes			// Value array
											+ num_elem_slots * num_IDs			// ID array
											+ bitcode_arr_size_num_IDTypes;		// Bitcode array

	return tot_arr_size_num_IDTypes;
}


template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::printRecur(std::ostream &os,
																			T *const tree_root,
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
