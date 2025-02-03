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

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcGlobalMemNeeded(const size_t num_elems, const unsigned threads_per_block)
{
	const size_t tree_arr_size_num_max_data_id_types = calcTreeArrSizeNumMaxDataIdTypes(num_elems, threads_per_block);

	size_t global_mem_needed = tree_arr_size_num_max_data_id_types;
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
		Space needed for instantiation = tree array size + addend, where addend = max(construction overhead, search overhead) = max(3 * num_elems * size of PointStructTemplate indices, num_elems * size of PointStructTemplate)
		Enough space to contain 3 size_t indices for every node is needed because the splitting of pointers in the dim2_val array at each node creates a need for the dim2_val arrays to be duplicated
		Space needed for reporting nodes is at most num_elems (if all elements are reported) * size of PointStructTemplate (if RetType = PointStructTemplate)
	*/
	const size_t construct_mem_overhead = num_elems * num_constr_working_arrs * sizeof(size_t);
	const size_t search_mem_max_overhead = num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>);
	global_mem_needed += (construct_mem_overhead > search_mem_max_overhead ? construct_mem_overhead : search_mem_max_overhead);

	return global_mem_needed;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeArrSizeNumMaxDataIDTypes(const size_t num_elems, const unsigned threads_per_block)
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
size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeSizeNumMaxDataIDTypes(const size_t num_elem_slots_per_tree)
{
	if constexpr (!HasID<PointStructTemplate<T, IDType, num_IDs>>::value)
		// No IDs present
		return calcTreeSizeNumTs<num_val_subarrs>(num_elem_slots_per_tree);
	else
	{
		// Separate size-comparison condition from the num_IDs==0 condition so that sizeof(IDType) is well-defined here, as often only one branch of a constexpr if is compiled
		if constexpr (sizeof(T) >= sizeof(IDType))
			// sizeof(T) >= sizeof(IDType), so calculate tree array size in units of sizeof(T) so that datatype T's alignment requirements will be satisfied
			return calcTreeSizeNumUs<T, num_val_subarrs, IDType, num_IDs>(num_elem_slots_per_tree);
		else
			// sizeof(IDType) > sizeof(T), so calculate total array size in units of sizeof(IDType) so that datatype IDType's alignment requirements will be satisfied
			return calcTreeSizeNumUs<IDType, num_IDs, T, num_val_subarrs>(num_elem_slots_per_tree);
	}
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <size_t num_T_subarrs>
size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeSizeNumTs(const size_t num_elem_slots_per_tree)
{
	/*
		tree_size_num_Ts = ceil(1/sizeof(T) * num_elem_slots_per_tree * (sizeof(T) * num_T_subarrs + 1 B/bitcode * 1 bitcode))
			With integer truncation:
				if tree_size_bytes % sizeof(T) != 0:
							= tree_size_bytes + 1
				if tree_size_bytes % sizeof(T) == 0:
							= tree_size_bytes
	*/
	// Calculate total size in bytes
	size_t tree_size_bytes = num_elem_slots_per_tree * (sizeof(T) * num_T_subarrs + 1);
	// Divide by sizeof(T)
	size_t tree_size_num_Ts = tree_size_bytes / sizeof(T);
	// If tree_size_bytes % sizeof(T) != 0, then tree_size_num_Ts * sizeof(T) < tree_size_bytes, so add 1 to tree_size_num_Ts
	if (tree_size_bytes % sizeof(T) != 0)
		tree_size_num_Ts++;
	return tree_size_num_Ts;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
template <typename U, size_t num_U_subarrs, typename V, size_t num_V_subarrs>
	requires SizeOfUAtLeastSizeOfV<U, V>
size_t StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::calcTreeSizeNumUs(const size_t num_elem_slots_per_tree)
{
	/*
		tree_size_num_Us = ceil(1/sizeof(U) * num_elem_slots_per_tree * (sizeof(U) * num_U_subarrs + sizeof(V) * num_V_subarrs + 1 B/bitcode * 1 bitcode))
			With integer truncation:
				if tree_size_bytes % sizeof(U) != 0:
							= tree_size_bytes + 1
				if tree_size_bytes % sizeof(U) == 0:
							= tree_size_bytes
	*/
	// Calculate total size in bytes
	size_t tree_size_bytes = num_elem_slots_per_tree * (sizeof(U) * num_U_subarrs + sizeof(V) * num_V_subarrs + 1);
	// Divide by sizeof(U)
	size_t tree_size_num_Us = tree_size_bytes / sizeof(U);
	// If tree_size_bytes % sizeof(U) != 0, then tree_size_num_Us * sizeof(U) < tree_size_bytes, so add 1 to tree_size_num_Us
	if (tree_size_bytes % sizeof(U) != 0)
		tree_size_num_Us++;
	return tree_size_num_Us;
}
