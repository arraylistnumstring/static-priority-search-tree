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
	: num_elem_slots_per_tree(calcNumElemSlotsPerTree(threads_per_block))
	num_elems(num_elems),
	// Total number of subtrees = num_thread_blocks = ceil(num_elems/threads_per_block)
	num_thread_blocks(num_elems / threads_per_block + (num_elems % threads_per_block == 0 ? 0 : 1)),
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
		num_elem_slots_per_tree = 0;
		return;
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
}
