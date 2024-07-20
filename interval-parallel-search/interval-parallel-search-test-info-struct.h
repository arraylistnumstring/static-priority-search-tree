#ifndef INTERVAL_PARALLEL_SEARCH_TEST_INFO_STRUCT_H
#define INTERVAL_PARALLEL_SEARCH_TEST_INFO_STRUCT_H

#include <algorithm>
#include <random>
#include <string>

#include "point-struct.h"

#include "interval-parallel-search-tester.h"


// Struct for information necessary to instantiate InterParaSearchTester
struct InterParaSearchTestInfoStruct
{
	// Ordering of fields chosen to minimise size of struct
	size_t rand_seed = 0;
	size_t num_elems;

	// Number of values necessary to define the bounds of an interval
	const static size_t NUM_VALS_INT_BOUNDS = 2;
	std::string val_range_strings[NUM_VALS_INT_BOUNDS] = {"0", "0"};

	std::string search_val_string;

	DataType data_type;

	DataType id_type;

	unsigned num_thread_blocks;
	unsigned threads_per_block;

	bool pts_with_ids = false;
	bool report_IDs = false;
	bool timed_CUDA = false;		// Whether to time CUDA code during testing

	// Instantiate outermost InterParaSearchTester type with respect to data type
	void test()
	{
		if (timed_CUDA)
		{
			InterParaSearchTester<true> ips_tester;
			
			dataTypeWrap(ips_tester);
		}
		else
		{
			InterParaSearchTester<false> ips_tester;

			dataTypeWrap(ips_tester);
		}
	};

	template <class InterParaSearchTesterTimingDet>
	void dataTypeWrap(InterParaSearchTesterTimingDet ips_tester)
	{
		if (data_type == DataType::DOUBLE)
		{
			typename InterParaSearchTesterTimingDet::DataTypeWrapper<double, std::uniform_real_distribution>
						ips_tester_data_type_instan(rand_seed, std::stod(val_range_strings[0]),
													std::stod(val_range_strings[1]),
													std::stod(search_val_string));
			
			numIDsWrap(ips_tester_data_type_instan);
		}
		else if (data_type == DataType::FLOAT)
		{
			typename InterParaSearchTesterTimingDet::DataTypeWrapper<float, std::uniform_real_distribution>
						ips_tester_data_type_instan(rand_seed, std::stod(val_range_strings[0]),
													std::stod(val_range_strings[1]),
													std::stod(search_val_string));
			
			numIDsWrap(ips_tester_data_type_instan);
		}
		else if (data_type == DataType::INT)
		{
			typename InterParaSearchTesterTimingDet::DataTypeWrapper<int, std::uniform_real_distribution>
						ips_tester_data_type_instan(rand_seed, std::stod(val_range_strings[0]),
													std::stod(val_range_strings[1]),
													std::stod(search_val_string));
			
			numIDsWrap(ips_tester_data_type_instan);
		}
		else if (data_type == DataType::LONG)
		{
			typename InterParaSearchTesterTimingDet::DataTypeWrapper<long, std::uniform_real_distribution>
						ips_tester_data_type_instan(rand_seed, std::stod(val_range_strings[0]),
													std::stod(val_range_strings[1]),
													std::stod(search_val_string));
			
			numIDsWrap(ips_tester_data_type_instan);
		}
	};

	// Instantiate next InterParaSearchTester type with respect to number of IDs
	template <typename InterParaSearchTesterDataTypesInstantiated>
	void numIDsWrap(InterParaSearchTesterDataTypesInstantiated ips_tester)
	{
		if (pts_with_ids)
		{
			typename InterParaSearchTesterDataTypesInstantiated::NumIDsWrapper<PointStruct, 1> ips_tester_num_ids_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 1 wrapper\n";
#endif
			
			IDTypeWrap(ips_tester_num_ids_instan);
		}
		else	// !pts_with_ids; can skip IDTypeWrap()
		{
			typename InterParaSearchTesterDataTypesInstantiated::NumIDsWrapper<PointStruct, 0> ips_tester_num_ids_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 0 wrapper\n";
#endif

			// As no ID distribution is used anyway, just place a dummy template template parameter taking one type parameter
			typename InterParaSearchTesterDataTypesInstantiated
						::NumIDsWrapper<PointStruct, 0>
						::IDTypeWrapper<std::uniform_real_distribution, void>
							ips_tester_fully_instan(ips_tester_num_ids_instan);

			testWrap(ips_tester_fully_instan);
		}
	};

	// Instantiate next InterParaSearchTester type with respect to ID type
	template <class InterParaSearchTesterDataTypeNumIDsInstantiated>
	void IDTypeWrap(InterParaSearchTesterDataTypeNumIDsInstantiated ips_tester)
	{
		if (pts_with_ids)
		{
			if (id_type == DataType::CHAR)
			{
				// typename necessary, as compiler defaults to treating nested names as variables
				typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, char> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = char wrapper\n";
#endif

				if (report_IDs)
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, char>::RetTypeWrapper<char> ips_tester_id_ret_types_instan(ips_tester_id_instan);

					testWrap(ips_tester_id_ret_types_instan);
				}
				else
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, char>::RetTypeWrapper ips_tester_id_ret_types_instan(ips_tester_id_instan);
					testWrap(ips_tester_id_ret_types_instan);
				}
			}
			else if (id_type == DataType::DOUBLE)
			{
				typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, double> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = double wrapper\n";
#endif

				if (report_IDs)
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, double>::RetTypeWrapper<double> ips_tester_id_ret_types_instan(ips_tester_id_instan);

					testWrap(ips_tester_id_ret_types_instan);
				}
				else
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, double>::RetTypeWrapper ips_tester_id_ret_types_instan(ips_tester_id_instan);
					testWrap(ips_tester_id_ret_types_instan);
				}
			}
			else if (id_type == DataType::FLOAT)
			{
				typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, float> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = float wrapper\n";
#endif

				if (report_IDs)
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, float>::RetTypeWrapper<float> ips_tester_id_ret_types_instan(ips_tester_id_instan);

					testWrap(ips_tester_id_ret_types_instan);
				}
				else
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, float>::RetTypeWrapper ips_tester_id_ret_types_instan(ips_tester_id_instan);
					testWrap(ips_tester_id_ret_types_instan);
				}
			}
			else if (id_type == DataType::INT)
			{
				typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, int> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = int wrapper\n";
#endif

				if (report_IDs)
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, int>::RetTypeWrapper<int> ips_tester_id_ret_types_instan(ips_tester_id_instan);

					testWrap(ips_tester_id_ret_types_instan);
				}
				else
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, int>::RetTypeWrapper ips_tester_id_ret_types_instan(ips_tester_id_instan);
					testWrap(ips_tester_id_ret_types_instan);
				}
			}
			else if (id_type == DataType::LONG)
			{
				typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, long> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
				std::cout << "Instantiated IDType = long wrapper\n";
#endif

				if (report_IDs)
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, long>::RetTypeWrapper<long> ips_tester_id_ret_types_instan(ips_tester_id_instan);

					testWrap(ips_tester_id_ret_types_instan);
				}
				else
				{
					typename InterParaSearchTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, long>::RetTypeWrapper ips_tester_id_ret_types_instan(ips_tester_id_instan);
					testWrap(ips_tester_id_ret_types_instan);
				}
			}
		}
	};

	// Run test
	template <typename InterParaSearchTesterClass>
	void testWrap(InterParaSearchTesterClass ips_tester)
	{
#ifdef DEBUG_WRAP
		std::cout << "Beginning test\n";
#endif

		ips_tester(num_elems, num_thread_blocks, threads_per_block);
	};
};


#endif
