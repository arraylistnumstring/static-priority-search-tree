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
	: num_elem_slots_per_tree(num_elems == 0 ? 0 : calcNumElemSlotsPerTree(threads_per_block)),
	num_elems(num_elems),
	// Each tree (except potentially the last one) contains as many elements as it can hold in order to reduce internal fragmentation and maximise thread occupancy
	// Total number of subtrees = num_thread_blocks = ceil(num_elems/threads_per_block)
	// As num_elem_slots_per_tree is declared before num_thread_blocks in the class declaration, it is also instantiated first
	// When this order is violated, no compilation error is reported; the data member that depends on a later-declared data member is simply incorrectly initialised
	num_thread_blocks(num_elems / num_elem_slots_per_tree + (num_elems % num_elem_slots_per_tree == 0 ? 0 : 1)),
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
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeSizeNumMaxDataIDTypes(const size_t num_elem_slots)
{
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		// No IDs present
		return calcTreeSizeNumTs<num_val_subarrs>(num_elem_slots);
	else
	{
		// Separate size-comparison condition from the num_IDs==0 condition so that sizeof(IDType) is well-defined here, as often only one branch of a constexpr if is compiled
		if constexpr (sizeof(T) >= sizeof(IDType))
			// sizeof(T) >= sizeof(IDType), so calculate tree array size in units of sizeof(T) so that datatype T's alignment requirements will be satisfied
			return calcTreeSizeNumUs<T, num_val_subarrs, IDType, num_IDs>(num_elem_slots);
		else
			// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
			return calcTreeSizeNumUs<IDType, num_IDs, T, num_val_subarrs>(num_elem_slots);
	}
}
