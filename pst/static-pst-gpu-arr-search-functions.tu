// Shared memory must be at least as large (number of threads) * (sizeof(long long) + sizeof(unsigned char))
// Non-member functions can only use at most one template clause
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
__global__ void twoSidedLeftSearchTreeArrGlobal(T *const tree_arr_d,
												const size_t full_tree_num_elem_slots,
												const size_t full_tree_size_num_max_data_id_types,
												RetType *const res_arr_d,
												const T max_dim1_val, const T min_dim2_val
											)
{
	/*
		By C++ specification 13.7.5 (Templates > Template declarations > Friends), point 9, "a friend function template with a constraint that depends on a template parameter from an enclosing template [is] a definition[...and] does not declare the same[...]function template as a declaration in any other scope."

		Moreover, a requires clause is part of a function's signature, such that the presence or lack of a requires clause changes the function which is called or referenced, so all requires clauses must be removed from the corresponding friend function.

		Finally, attempting to use a requires clause for differently named template parameters (e.g. when declaring a friend template) causes a multiple-overload compilation failure, even when such parameters would have evaluated to equivalent dependencies and function signatures.

		Hence, the static_assert here takes the place of the equivalent requires clause.
	*/
	static_assert(std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
					>::value,
					"RetType is not of type PointStructTemplate<T, IDType, num_IDs>, nor of type IDType");

	// Use char datatype because extern variables must be consistent across all declarations and because char is the smallest possible datatype
	extern __shared__ char s[];
	// Node indices for each thread to search
	long long *search_inds_arr = reinterpret_cast<long long *>(s);
	unsigned char *search_codes_arr = reinterpret_cast<unsigned char *>(search_inds_arr + blockDim.x);
	// Initialise shared memory
	// All threads except for thread 0 start by being inactive
	if (threadIdx.x == 0)
		search_inds_arr[threadIdx.x] = 0;
	else
		search_inds_arr[threadIdx.x] = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::IndexCodes::INACTIVE_IND;

	// For twoSidedLeftSearchTreeArrGlobal(), all threads start with their search code set to LEFT_SEARCH
	search_codes_arr[threadIdx.x] = StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::SearchCodes::LEFT_SEARCH;

	__syncthreads();	// Must synchronise before processing to ensure data is properly set
}
