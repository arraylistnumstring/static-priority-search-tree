#ifndef PST_TESTER_H
#define PST_TESTER_H

#include <algorithm>	// To use sort()
#include <iostream>
#include <random>		// To use std::mt19937
#include <type_traits>

#include "../err-chk.h"
#include "../print-array.h"

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
			typename RandNumEng=std::mt19937>
	requires std::is_arithmetic<T>::value
struct PSTTester
{
	RandNumEng rand_num_eng;
	Distrib<T> distr;

	// Search values to use; may not be used if test is not of search type
	T dim1_val_bound1;
	T dim1_val_bound2;
	T min_dim2_val;

	PSTTester(T min_val, T max_val, T dim1_val_bound1, T dim1_val_bound2, T min_dim2_val)
		: rand_num_eng(0),
		distr(min_val, max_val),
		dim1_val_bound1(dim1_val_bound1),
		dim1_val_bound2(dim1_val_bound2),
		min_dim2_val(min_dim2_val)
	{};

	PSTTester(size_t rand_seed, T min_val, T max_val, T dim1_val_bound1, T dim1_val_bound2,
				T min_dim2_val)
		: rand_num_eng(rand_seed),
		distr(min_val, max_val),
		dim1_val_bound1(dim1_val_bound1),
		dim1_val_bound2(dim1_val_bound2),
		min_dim2_val(min_dim2_val)
	{};

	// Functor to allow for templated function calling without needing to pass in dummy parameters just to indicate type
	// Nested structs to allow for the metaprogramming equivalent of currying, but with types instead of parameters
	template <template<typename, typename, size_t> class PointStructTemplate,
				template<typename, template<typename, typename, size_t> class, typename, size_t> class StaticPSTTemplate>
	struct TreeTypeWrapper
	{
		// Allow IDType and num_IDs to be determined separately from PointStructTemplate and StaticPSTTemplate; essentially currying at the template level with types
		template <typename IDType>
		struct IDTypeWrapper
		{
			template <size_t num_IDs>
			struct NumIDsWrapper
			{
				void operator()(size_t num_elems, PSTTestCodes test_type=CONSTRUCT)
				{
					PointStructTemplate<T, IDType, num_IDs> *pt_arr = new PointStructTemplate<T, IDType, num_IDs>[num_elems];

					for (size_t i = 0; i < num_elems; i++)
					{
						// Distribution takes random number engine as parameter with which to generate its next value
						pt_arr[i].dim1_val = distr(rand_num_eng);
						pt_arr[i].dim2_val = distr(rand_num_eng);
						// Lazy instantiation of value of type IDType from type T
						if constexpr (num_IDs == 1)
							pt_arr[i].id = *const_cast<IDType *>(&distr(rand_num_eng));
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
					PointStructTemplate<T, IDType, num_IDs> *res_pt_arr;

					// Search/report test phase
					if (test_type == LEFT_SEARCH)
					{
						res_pt_arr = tree->twoSidedLeftSearch(num_res_elems, dim1_val_bound1, min_dim2_val);
					}
					else if (test_type == RIGHT_SEARCH)
					{
						res_pt_arr = tree->twoSidedRightSearch(num_res_elems, dim1_val_bound1, min_dim2_val);
					}
					else if (test_type == THREE_SEARCH)
					{
						res_pt_arr = tree->threeSidedSearch(num_res_elems, dim1_val_bound1, dim1_val_bound2, min_dim2_val);
					}
					// If test_type == CONSTRUCT, do nothing for the search/report phase

					std::sort(res_pt_arr, res_pt_arr + num_res_elems,
							[](const PointStructTemplate<T, IDType, num_IDs> &pt_1,
								const PointStructTemplate<T, IDType, num_IDs> &pt_2)
							{
							return pt_1.compareDim1(pt_2) < 0;
							});

					printArray(std::cout, res_pt_arr, 0, num_res_elems);
					std::cout << '\n';

					delete tree;
					delete[] pt_arr;
				};
			};
		};
	};
};


#endif
