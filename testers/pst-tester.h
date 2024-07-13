#ifndef PST_TESTER_H
#define PST_TESTER_H

#include <algorithm>	// To use sort()
#include <iostream>
#include <limits>		// To get numeric limits of each datatype
#include <random>		// To use std::mt19937
#include <type_traits>

#include "err-chk.h"
#include "gpu-err-chk.h"
#include "print-array.h"

enum DataType {CHAR, DOUBLE, FLOAT, INT, LONG};

enum PSTTestCodes
{
	CONSTRUCT,
	LEFT_SEARCH,
	RIGHT_SEARCH,
	THREE_SEARCH
};

enum PSTType {CPU_ITER, CPU_RECUR, GPU};


template <typename T,
			template<typename> typename Distrib,
			// Default random number engine: Mersenne Twister 19937; takes its constructor parameter as its seed
			typename RandNumEng=std::mt19937,
			bool timed_CUDA=false>
	requires std::is_arithmetic<T>::value
struct PSTTester
{
	RandNumEng rand_num_eng;
	Distrib<T> distr;

	// Search values to use; may not be used if test is not of search type
	T dim1_val_bound1;
	T dim1_val_bound2;
	T min_dim2_val;

	bool vals_inc_ordered;

	PSTTester(T min_val, T max_val, T dim1_val_bound1, T dim1_val_bound2, T min_dim2_val, bool vals_inc_ordered)
		: rand_num_eng(0),
		distr(min_val, max_val),
		dim1_val_bound1(dim1_val_bound1),
		dim1_val_bound2(dim1_val_bound2),
		min_dim2_val(min_dim2_val),
		vals_inc_ordered(vals_inc_ordered)
	{};

	PSTTester(size_t rand_seed, T min_val, T max_val, T dim1_val_bound1, T dim1_val_bound2,
				T min_dim2_val, bool vals_inc_ordered)
		: rand_num_eng(rand_seed),
		distr(min_val, max_val),
		dim1_val_bound1(dim1_val_bound1),
		dim1_val_bound2(dim1_val_bound2),
		min_dim2_val(min_dim2_val),
		vals_inc_ordered(vals_inc_ordered)
	{};

	// Nested structs to allow for the metaprogramming equivalent of currying, but with type parameters
	template <template<typename, typename, size_t> class PointStructTemplate,
				template<typename, template<typename, typename, size_t> class, typename, size_t> class StaticPSTTemplate>
	struct TreeTypeWrapper
	{
		// Nested class have access to all levels of access of their enclosing scope; however, as nested classes are not associated with any enclosing class instance in particular, so must keep track of the desired "parent" instance, if any
		PSTTester<T, Distrib, RandNumEng> pst_tester;

		TreeTypeWrapper(PSTTester<T, Distrib, RandNumEng> pst_tester)
			: pst_tester(pst_tester)
		{};

		template <size_t num_IDs>
		struct NumIDsWrapper
		{
			TreeTypeWrapper<PointStructTemplate, StaticPSTTemplate> tree_type_wrapper;
			
			NumIDsWrapper(TreeTypeWrapper<PointStructTemplate, StaticPSTTemplate> tree_type_wrapper)
				: tree_type_wrapper(tree_type_wrapper)
			{};

			template <template <typename> typename IDDistrib, typename IDType, typename RetType>
				// Requires that RetType is either of type IDType or of type PointStructTemplate<T, IDType, num_IDs>
				// std::disjunction<B1, ..., Bn> performs a logical OR on enclosed type traits, specifically on the value returned by \Vee_{i=1}^n bool(B1::value)
				requires std::disjunction<
					std::is_same<RetType, IDType>,
									std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
                        >::value
			struct IDTypeWrapper
			{
				NumIDsWrapper<num_IDs> num_ids_wrapper;

				// Bounds of distribution [a, b] must satisfy b - a <= std::numeric_limits<IDType>::max()
				IDDistrib<IDType> id_distr;

				IDTypeWrapper(NumIDsWrapper<num_IDs> num_ids_wrapper)
					: num_ids_wrapper(num_ids_wrapper),
					id_distr(0, std::numeric_limits<IDType>::max())
				{};

				void operator()(size_t num_elems, PSTTestCodes test_type=CONSTRUCT)
				{
					PointStructTemplate<T, IDType, num_IDs> *pt_arr = new PointStructTemplate<T, IDType, num_IDs>[num_elems];

					for (size_t i = 0; i < num_elems; i++)
					{
						// Distribution takes random number engine as parameter with which to generate its next value
						T val1 = num_ids_wrapper.tree_type_wrapper.pst_tester.distr(num_ids_wrapper.tree_type_wrapper.pst_tester.rand_num_eng);
						T val2 = num_ids_wrapper.tree_type_wrapper.pst_tester.distr(num_ids_wrapper.tree_type_wrapper.pst_tester.rand_num_eng);

						// Swap generated values only if val1 > val2 and monotonically increasing order is required
						if (num_ids_wrapper.tree_type_wrapper.pst_tester.vals_inc_ordered
								&& val1 > val2)
						{
							pt_arr[i].dim1_val = val2;
							pt_arr[i].dim2_val = val1;
						}
						else
						{
							pt_arr[i].dim1_val = val1;
							pt_arr[i].dim2_val = val2;
						}
						// Instantiation of value of type IDType
						if constexpr (num_IDs == 1)
							pt_arr[i].id = id_distr(num_ids_wrapper.tree_type_wrapper.pst_tester.rand_num_eng);
					}

#ifdef DEBUG
					printArray(std::cout, pt_arr, 0, num_elems);
					std::cout << '\n';
#endif


					StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs> *tree =
						new StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>(pt_arr, num_elems);

					if (tree == nullptr)
					{
						throwErr("Error: Could not allocate memory for priority search tree");
						return;
					}

					std::cout << *tree << '\n';

					size_t num_res_elems = 0;
					RetType *res_arr;

					if constexpr (timed_CUDA)
						// Start CUDA timer

					// Search/report test phase
					if (test_type == LEFT_SEARCH)
					{
						res_arr = tree->twoSidedLeftSearch(num_res_elems,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
					}
					else if (test_type == RIGHT_SEARCH)
					{
						res_arr = tree->twoSidedRightSearch(num_res_elems,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
					}
					else if (test_type == THREE_SEARCH)
					{
						res_arr = tree->threeSidedSearch(num_res_elems,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound2,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
					}
					// If test_type == CONSTRUCT, do nothing for the search/report phase
					
					if constexpr (timed_CUDA)
						// End CUDA timer
						// Report timing

					// If result pointer array is on GPU, copy it to CPU and print
					cudaPointerAttributes ptr_info;
					gpuErrorCheck(cudaPointerGetAttributes(&ptr_info, res_arr),
									"Error in determining location type of memory address of result RetType array (i.e. whether on host or device)");

					// res_arr is on device; copy to CPU
					if (ptr_info.type == cudaMemoryTypeDevice)
					{
						// Differentiate on-device and on-host pointers
						PointStructTemplate<T, IDType, num_IDs> *res_arr_d = res_arr;

						res_arr = nullptr;

						// Allocate space on host for data
						res_arr = new RetType[num_res_elems];

						if (res_arr == nullptr)
							throwErr("Error: could not allocate RetType array of size "
										+ std::to_string(num_res_elems) + " on host");

						// Copy data from res_arr_d to res_arr
						gpuErrorCheck(cudaMemcpy(res_arr, res_arr_d, num_res_elems * sizeof(RetType),
											cudaMemcpyDefault), 
										"Error in copying array of RetType objects from device "
										+ std::to_string(ptr_info.device) + ": ");

						// Free on-device array of PointStructTemplates
						gpuErrorCheck(cudaFree(res_arr_d), "Error in freeing on-device array of result RetType objects on device "
										+ std::to_string(ptr_info.device) + ": ");
					}

					// Sort output for consistency (specifically compared to GPU-reported outputs, which may be randomly ordered and must therefore be sorted for easy comparisons)
					if constexpr (std::is_same<RetType, IDType>::value)
						std::sort(res_arr, res_arr + num_res_elems,
								[](const IDType &id_1, const IDType &id_2)
								{
									return id_1 < id_2;
								});
					else
						std::sort(res_arr, res_arr + num_res_elems,
								[](const PointStructTemplate<T, IDType, num_IDs> &pt_1,
									const PointStructTemplate<T, IDType, num_IDs> &pt_2)
								{
									return pt_1.compareDim1(pt_2) < 0;
								});

					// For some reason, the offending line is the access of ptstr.print()
					printArray(std::cout, res_arr, 0, num_res_elems);
					std::cout << '\n';

					delete tree;
					delete[] pt_arr;
					delete[] res_arr;
				};
			};

			// Template specialisation for case with no ID and therefore no ID distribution; sepcialisation must follow primary (completely unspecified) template; full specialisation not allowed in class scope, hence the remaining dummy type
			template <template <typename> typename IDDistrib>
			struct IDTypeWrapper<IDDistrib, void>
			{
				NumIDsWrapper<num_IDs> num_ids_wrapper;

				IDTypeWrapper(NumIDsWrapper<num_IDs> num_ids_wrapper)
					: num_ids_wrapper(num_ids_wrapper)
				{};

				void operator()(size_t num_elems, PSTTestCodes test_type=CONSTRUCT)
				{
					PointStructTemplate<T, void, num_IDs> *pt_arr = new PointStructTemplate<T, void, num_IDs>[num_elems];

					for (size_t i = 0; i < num_elems; i++)
					{
						// Distribution takes random number engine as parameter with which to generate its next value
						T val1 = num_ids_wrapper.tree_type_wrapper.pst_tester.distr(num_ids_wrapper.tree_type_wrapper.pst_tester.rand_num_eng);
						T val2 = num_ids_wrapper.tree_type_wrapper.pst_tester.distr(num_ids_wrapper.tree_type_wrapper.pst_tester.rand_num_eng);

						// Swap generated values only if val1 > val2 and monotonically increasing order is required
						if (num_ids_wrapper.tree_type_wrapper.pst_tester.vals_inc_ordered
								&& val1 > val2)
						{
							pt_arr[i].dim1_val = val2;
							pt_arr[i].dim2_val = val1;
						}
						else
						{
							pt_arr[i].dim1_val = val1;
							pt_arr[i].dim2_val = val2;
						}
					}

#ifdef DEBUG
					printArray(std::cout, pt_arr, 0, num_elems);
					std::cout << '\n';
#endif


					StaticPSTTemplate<T, PointStructTemplate, void, num_IDs> *tree =
						new StaticPSTTemplate<T, PointStructTemplate, void, num_IDs>(pt_arr, num_elems);

					if (tree == nullptr)
					{
						throwErr("Error: Could not allocate memory for priority search tree");
						return;
					}

					std::cout << *tree << '\n';

					size_t num_res_elems = 0;
					PointStructTemplate<T, void, num_IDs> *res_pt_arr;

					if constexpr (timed_CUDA)
						// Start CUDA timer

					// Search/report test phase
					if (test_type == LEFT_SEARCH)
					{
						res_pt_arr = tree->twoSidedLeftSearch(num_res_elems,
																num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
																num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
					}
					else if (test_type == RIGHT_SEARCH)
					{
						res_pt_arr = tree->twoSidedRightSearch(num_res_elems,
																num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
																num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
					}
					else if (test_type == THREE_SEARCH)
					{
						res_pt_arr = tree->threeSidedSearch(num_res_elems,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound2,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
					}
					// If test_type == CONSTRUCT, do nothing for the search/report phase

					if constexpr (timed_CUDA)
						// End CUDA timer
						// Report timing

					// If result pointer array is on GPU, copy it to CPU and print
					cudaPointerAttributes ptr_info;
					gpuErrorCheck(cudaPointerGetAttributes(&ptr_info, res_pt_arr),
									"Error in determining location type of memory address of result PointStruct array (i.e. whether on host or device)");

					// res_pt_arr is on device; copy to CPU
					if (ptr_info.type == cudaMemoryTypeDevice)
					{
						// Differentiate on-device and on-host pointers
						PointStructTemplate<T, void, num_IDs> *res_pt_arr_d = res_pt_arr;

						res_pt_arr = nullptr;

						// Allocate space on host for data
						res_pt_arr = new PointStructTemplate<T, void, num_IDs>[num_res_elems];

						if (res_pt_arr == nullptr)
							throwErr("Error: could not allocate PointStructTemplate<T, IDType, num_IDs> array of size "
										+ std::to_string(num_res_elems) + " on host");

						// Copy data from res_pt_arr_d to res_pt_arr
						gpuErrorCheck(cudaMemcpy(res_pt_arr, res_pt_arr_d, num_res_elems * sizeof(PointStructTemplate<T, void, num_IDs>),
											cudaMemcpyDefault), 
										"Error in copying array of PointStructTemplate<T, void, num_IDs> objects from device "
										+ std::to_string(ptr_info.device) + ": ");

						// Free on-device array of PointStructTemplates
						gpuErrorCheck(cudaFree(res_pt_arr_d), "Error in freeing on-device array of result PointStructs on device "
										+ std::to_string(ptr_info.device) + ": ");
					}

					// Sort output for consistency (specifically compared to GPU-reported outputs, which may be randomly ordered and must therefore be sorted for easy comparisons)
					std::sort(res_pt_arr, res_pt_arr + num_res_elems,
							[](const PointStructTemplate<T, void, num_IDs> &pt_1,
								const PointStructTemplate<T, void, num_IDs> &pt_2)
							{
								return pt_1.compareDim1(pt_2) < 0;
							});

					// For some reason, the offending line is the access of ptstr.print()
					printArray(std::cout, res_pt_arr, 0, num_res_elems);
					std::cout << '\n';

					delete tree;
					delete[] pt_arr;
					delete[] res_pt_arr;
				};
			};
		};
	};
};


#endif
