#include <string>						// To use string-building functions
#include <thrust/execution_policy.h>	// To use thrust::cuda::par::on() stream-specifying execution policy for sorting
#include <thrust/sort.h>				// To use parallel sorting algorithm

#include "arr-ind-assign.h"
#include "class-member-checkers.h"

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::StaticPSTGPUArr(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d,
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

	gpuErrorCheck(cudaEventRecord(populate_tree_start),
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
						 pt_arr_d, dim1_val_ind_arr_d, dim2_val_ind_arr_d,
						 dim2_val_ind_arr_secondary_d, num_elems);
	}
	else
	{
		// sizeof(IDType) > sizeof(T), and the latter is guaranteed to be a factor of the former
		populateTrees<<<num_thread_blocks, threads_per_block,
							threads_per_block * sizeof(size_t) * num_constr_working_arrs>>>
						(tree_arr_d, full_tree_num_elem_slots,
						 full_tree_size_num_max_data_id_types * sizeof(IDType) / sizeof(T),
						 pt_arr_d, dim1_val_inD_arr_d, dim2_val_ind_arr_d,
						 dim2_val_ind_arr_secondary_d, num_elems);
	}
}


// const keyword after method name indicates that the method does not modify any data members of the associated class
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
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
	global_mem_needed += (construct_mem_overhead > search_mem_max_overhead ? construct_mem_overhead : search_mem_max_overhead);

	return global_mem_needed;
}


// Construction-related helper functions

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
__forceinline__ __device__ long long StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::binarySearch(
															PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d,
															size_t *const &dim1_val_ind_arr_d,
															PointStructTemplate<T, IDType, num_IDs> &elem_to_find,
															const size_t &init_ind,
															const size_t &num_elems
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
