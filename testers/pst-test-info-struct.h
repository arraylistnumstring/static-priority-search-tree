#ifndef PST_TEST_INFO_STRUCT_H
#define PST_TEST_INFO_STRUCT_H

#include <algorithm>
#include <random>
#include <string>

#include "point-struct.h"
#include "pst-tester.h"
#include "static-pst-cpu-iter.h"
#include "static-pst-cpu-recur.h"
#include "static-pst-gpu.h"


// Struct for information necessary to instantiate PSTTester
struct PSTTestInfoStruct
{
	enum NumSearchVals
	{
		NUM_VALS_TWO_SEARCH=2,
		NUM_VALS_THREE_SEARCH=3
	};
	DataType data_type;
	PSTTestCodes test_type;
	PSTType tree_type;
	std::string search_range_strings[NumSearchVals::NUM_VALS_THREE_SEARCH] = {"0", "0", "0"};

	bool pts_with_ids = false;
	DataType id_type;

	size_t rand_seed = 0;

	// Number of values necessary to define the bounds of an interval
	const static size_t NUM_VALS_INT_BOUNDS = 2;
	std::string tree_val_range_strings[NUM_VALS_INT_BOUNDS] = {"0", "0"};

	size_t num_elems;

	// Instantiate outermost PSTTester type with respect to data type
	void test()
	{
		if (data_type == DataType::DOUBLE)
		{
			PSTTester<double, std::uniform_real_distribution>
						pst_tester(rand_seed, std::stod(tree_val_range_strings[0]),
									std::stod(tree_val_range_strings[1]),
									std::stod(search_range_strings[0]),
									std::stod(search_range_strings[1]),
									std::stod(search_range_strings[2]));
			
			treeTypeWrap(pst_tester);
		}
		else if (data_type == DataType::FLOAT)
		{
			PSTTester<float, std::uniform_real_distribution>
						pst_tester(rand_seed, std::stof(tree_val_range_strings[0]),
									std::stof(tree_val_range_strings[1]),
									std::stof(search_range_strings[0]),
									std::stof(search_range_strings[1]),
									std::stof(search_range_strings[2]));
			
			treeTypeWrap(pst_tester);
		}
		else if (data_type == DataType::INT)
		{
			PSTTester<int, std::uniform_int_distribution>
						pst_tester(rand_seed, std::stoi(tree_val_range_strings[0]),
									std::stoi(tree_val_range_strings[1]),
									std::stoi(search_range_strings[0]),
									std::stoi(search_range_strings[1]),
									std::stoi(search_range_strings[2]));
			
			treeTypeWrap(pst_tester);
		}
		else if (data_type == DataType::LONG)
		{
			PSTTester<long, std::uniform_int_distribution>
						pst_tester(rand_seed, std::stol(tree_val_range_strings[0]),
									std::stol(tree_val_range_strings[1]),
									std::stol(search_range_strings[0]),
									std::stol(search_range_strings[1]),
									std::stol(search_range_strings[2]));
			
			treeTypeWrap(pst_tester);
		}
	};

	// Instantiate next PSTTester type with respect to tree type
	template <typename PSTTesterDataIDInfoInstantiated>
	void treeTypeWrap(PSTTesterDataIDInfoInstantiated pst_tester)
	{
		if (tree_type == PSTType::CPU_ITER)
		{
			typename PSTTesterDataIDInfoInstantiated::TreeTypeWrapper<PointStruct, StaticPSTCPUIter> pst_tester_tree_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated StaticPSTCPUIter wrapper\n";
#endif

			numIDsWrap(pst_tester_tree_instan);
		}
		else if (tree_type == PSTType::CPU_RECUR)
		{
			typename PSTTesterDataIDInfoInstantiated::TreeTypeWrapper<PointStruct, StaticPSTCPURecur> pst_tester_tree_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated StaticPSTCPURecur wrapper\n";
#endif

			numIDsWrap(pst_tester_tree_instan);
		}
		else	// tree_type == PSTType::GPU
		{
			typename PSTTesterDataIDInfoInstantiated::TreeTypeWrapper<PointStruct, StaticPSTGPU> pst_tester_tree_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated StaticPSTGPU wrapper\n";
#endif

			numIDsWrap(pst_tester_tree_instan);
		}
	};

	// Instantiate next PSTTester type with respect to number of IDs
	template <typename PSTTesterDataIDTypesInstantiated>
	void numIDsWrap(PSTTesterDataIDTypesInstantiated pst_tester)
	{
		if (pts_with_ids)
		{
			typename PSTTesterDataIDTypesInstantiated::NumIDsWrapper<1> pst_tester_num_ids_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 1 wrapper\n";
#endif
			
			IDTypeWrap(pst_tester_num_ids_instan);
		}
		else	// !pts_with_ids; can skip IDTypeWrap()
		{
			typename PSTTesterDataIDTypesInstantiated::NumIDsWrapper<0> pst_tester_num_ids_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 0 wrapper\n";
#endif

			// As no ID distribution is used anyway, just place a dummy template template parameter taking one type parameter
			typename PSTTesterDataIDTypesInstantiated
						::NumIDsWrapper<0>
						::IDTypeWrapper<std::uniform_real_distribution, void>
							pst_tester_fully_instan(pst_tester_num_ids_instan);

			testWrap(pst_tester_fully_instan);
		}
	};

	// Instantiate next PSTTester type with respect to ID type
	template <class PSTTesterDataInstantiated>
	void IDTypeWrap(PSTTesterDataInstantiated pst_tester)
	{
		if (pts_with_ids)
		{
			if (id_type == DataType::CHAR)
			{
				// typename necessary, as compiler defaults to treating nested names as variables
				typename PSTTesterDataInstantiated::IDTypeWrapper<std::uniform_int_distribution, char> pst_tester_id_instan(pst_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = char wrapper\n";
#endif

				testWrap(pst_tester_id_instan);
			}
			else if (id_type == DataType::DOUBLE)
			{
				typename PSTTesterDataInstantiated::IDTypeWrapper<std::uniform_real_distribution, double> pst_tester_id_instan(pst_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = double wrapper\n";
#endif

				testWrap(pst_tester_id_instan);
			}
			else if (id_type == DataType::FLOAT)
			{
				typename PSTTesterDataInstantiated::IDTypeWrapper<std::uniform_real_distribution, float> pst_tester_id_instan(pst_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = float wrapper\n";
#endif

				testWrap(pst_tester_id_instan);
			}
			else if (id_type == DataType::INT)
			{
				typename PSTTesterDataInstantiated::IDTypeWrapper<std::uniform_int_distribution, int> pst_tester_id_instan(pst_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = int wrapper\n";
#endif

				testWrap(pst_tester_id_instan);
			}
			else if (id_type == DataType::LONG)
			{
				typename PSTTesterDataInstantiated::IDTypeWrapper<std::uniform_int_distribution, long> pst_tester_id_instan(pst_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = long wrapper\n";
#endif

				testWrap(pst_tester_id_instan);
			}
		}
	};

	// Run test
	template <typename PSTTesterClass>
	void testWrap(PSTTesterClass pst_tester)
	{
#ifdef DEBUG_WRAP
		std::cout << "Beginning test\n";
#endif

		pst_tester(num_elems, test_type);
	};
};


#endif
