#ifndef PST_TEST_INFO_STRUCT_H
#define PST_TEST_INF_STRUCT_H

#include <random>
#include <string>

#include "../point-struct.h"
#include "../static-pst-cpu-iter.h"
#include "../static-pst-cpu-recur.h"
#include "../static-pst-gpu.h"
#include "pst-tester.h"


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
	std::string search_range_strings[NumSearchVals::NUM_VALS_THREE_SEARCH];

	bool pts_with_ids = false;
	DataType id_type;

	size_t rand_seed = 0;

	// Number of values necessary to define the bounds of an interval
	const static size_t NUM_VALS_INT_BOUNDS = 2;
	std::string tree_val_range_strings[NUM_VALS_INT_BOUNDS];

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
			
			IDTypeWrap(pst_tester);
		}
		else if (data_type == DataType::FLOAT)
		{
			PSTTester<float, std::uniform_real_distribution>
						pst_tester(rand_seed, std::stof(tree_val_range_strings[0]),
									std::stof(tree_val_range_strings[1]),
									std::stof(search_range_strings[0]),
									std::stof(search_range_strings[1]),
									std::stof(search_range_strings[2]));
			
			IDTypeWrap(pst_tester);
		}
		else if (data_type == DataType::INT)
		{
			PSTTester<int, std::uniform_int_distribution>
						pst_tester(rand_seed, std::stoi(tree_val_range_strings[0]),
									std::stoi(tree_val_range_strings[1]),
									std::stoi(search_range_strings[0]),
									std::stoi(search_range_strings[1]),
									std::stoi(search_range_strings[2]));
			
			IDTypeWrap(pst_tester);
		}
		else if (data_type == DataType::LONG)
		{
			PSTTester<long, std::uniform_int_distribution>
						pst_tester(rand_seed, std::stol(tree_val_range_strings[0]),
									std::stol(tree_val_range_strings[1]),
									std::stol(search_range_strings[0]),
									std::stol(search_range_strings[1]),
									std::stol(search_range_strings[2]));
			
			IDTypeWrap(pst_tester);
		}
	};

	// Instantiate next PSTTester type with respect to ID type
	template <class PSTTesterDataInstantiated>
	void IDTypeWrap(PSTTesterDataInstantiated pst_tester)
	{
		if (id_type == DataType::CHAR)
		{
			typename PSTTesterDataInstantiated::IDTypeWrapper<double> pst_tester_id_instan(pst_tester);

			numIDsWrap(pst_tester_id_instan);
		}
		else if (data_type == DataType::DOUBLE)
		{
			pst_tester.IDTypeWrapper<double> pst_tester_id_instan;

			numIDsWrap(pst_tester_id_instan);
		}
		else if (data_type == DataType::FLOAT)
		{
			pst_tester.IDTypeWrapper<float> pst_tester_id_instan;

			numIDsWrap(pst_tester_id_instan);
		}
		else if (data_type == DataType::INT)
		{
			pst_tester.IDTypeWrapper<int> pst_tester_id_instan;

			numIDsWrap(pst_tester_id_instan);
		}
		else if (data_type == DataType::LONG)
		{
			pst_tester.IDTypeWrapper<long> pst_tester_id_instan;

			numIDsWrap(pst_tester_id_instan);
		}
	};

	// Instantiate next PSTTester type with respect to number of IDs
	template <typename PSTTesterDataIDTypesInstantiated>
	void numIDsWrap(PSTTesterDataIDTypesInstantiated pst_tester)
	{
		if (pts_with_ids)
		{
			pst_tester.NumIDsWrapper<1> pst_tester_num_ids_instan;
			
			treeTypeWrap(pst_tester_num_ids_instan);
		}
		else	// !pts_with_ids
		{
			pst_tester.NumIDsWrapper<0> pst_tester_ids_instan;

			treeTypeWrap(pst_tester_num_ids_instan);
		}
	};

	// Instantiate next PSTTester type with respect to tree type
	template <typename PSTTesterDataIDInfoInstantiated>
	void treeTypeWrap(PSTTesterDataIDInfoInstantiated pst_tester)
	{
		if (tree_type == PSTType::CPU_ITER)
		{
			pst_tester.TreeTypeWrapper<PointStruct, StaticPSTCPUIter> pst_tester_tree_instan;

			testWrap(pst_tester_tree_instan);
		}
		else if (tree_type == PSTType::CPU_RECUR)
		{
			pst_tester.TreeTypeWrapper<PointStruct, StaticPSTCPURecur> pst_tester_tree_instan;

			testWrap(pst_tester_tree_instan);
		}
		else	// tree_type == PSTType::GPU
		{
			pst_tester.TreeTypeWrapper<PointStruct, StaticPSTGPU> pst_tester_tree_instan;

			testWrap(pst_tester_tree_instan);
		}
	};

	// Run test
	template <typename PSTTesterClass>
	void testWrap(PSTTesterClass pst_tester)
	{
		pst_tester(num_elems, test_type);
	};
};


#endif
