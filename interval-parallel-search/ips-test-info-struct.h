#ifndef IPS_TEST_INFO_STRUCT_H
#define IPS_TEST_INFO_STRUCT_H

#include <algorithm>
#include <cstdlib>		// To use std::exit()
#include <random>
#include <string>

#include "exit-status-codes.h"
#include "ips-tester.h"
#include "preprocessor-symbols.h"
#include "point-struct.h"


// Struct for information necessary to instantiate IPSTester
struct IPSTestInfoStruct
{
	// Ordering of fields chosen to minimise size of struct; std::string type appears to take 32 bytes

	std::string input_file = "";

	std::string pt_grid_dim_strings[NUM_DIMS] = {"0", "0", "0"};
	std::string metacell_dim_strings[NUM_DIMS] = {"64", "0", "0"};

	std::string search_val_string;

	// Number of values necessary to define the bounds of an interval
	const static size_t MIN_NUM_VALS_INT_BOUNDS = 2;
	const static size_t MAX_NUM_VALS_INT_BOUNDS = 4;
	std::string val_range_strings[MAX_NUM_VALS_INT_BOUNDS] = {"0", "0", "-1", "-1"};

	size_t rand_seed = 0;
	size_t num_elems;

	// Data types are chosen to correspond to CUDA data types for corresponding on-device values
	// Default values correspond with those chosen by Liu et al. in their original paper
	unsigned num_thread_blocks;
	unsigned threads_per_block = 128;

	// By the standard, enums must be capable of holding int values, though the actual data-type can be char, signed int or unsigned int, as long as the chosen type can hold all values in the enumeration
	DataType data_type;
	DataType id_type;

	bool pts_with_ids = false;
	bool report_IDs = false;
	bool timed_CUDA = false;		// Whether to time CUDA code during testing

	// Instantiate outermost IPSTester type with respect to CUDA timing
	void test()
	{
		// Must explicitly set class instantiation variables to true and false in each branch in order for code to be compile-time determinable
		if (timed_CUDA)
			dataTypeWrap<IPSTester<true>>();
		else
			dataTypeWrap<IPSTester<false>>();
	};

	template <class IPSTesterTimingDet>
	void dataTypeWrap()
	{
		if (data_type == DataType::DOUBLE)
			// typename necessary, as compiler defaults to treating nested names as variables
			// Function pointers cannot have default arguments (and presumably std::functions operate the same way, at least when passed as parameters), so wrap std::stod in a lambda function with the correct signature
			// Casting necessary as compiler fails to recognise a lambda as an std::function of the same signature and return type, thereby failing to instantiate template function
			numIDsWrapCaller<typename IPSTesterTimingDet::DataTypeWrapper<double, std::uniform_real_distribution>>(static_cast<std::function<double(const std::string &)>>(
						[](const std::string &str) -> double
						{
							return std::stod(str);
						})
					);
		else if (data_type == DataType::FLOAT)
			numIDsWrapCaller<typename IPSTesterTimingDet::DataTypeWrapper<float, std::uniform_real_distribution>>(static_cast<std::function<float(const std::string &)>>(
						[](const std::string &str) -> float
						{
							return std::stof(str);
						})
					);
		else if (data_type == DataType::INT)
			numIDsWrapCaller<typename IPSTesterTimingDet::DataTypeWrapper<int, std::uniform_int_distribution>>(static_cast<std::function<int(const std::string &)>>(
						[](const std::string &str) -> int
						{
							return std::stoi(str);
						})
					);
		else if (data_type == DataType::LONG)
			numIDsWrapCaller<typename IPSTesterTimingDet::DataTypeWrapper<long, std::uniform_int_distribution>>(static_cast<std::function<long(const std::string &)>>(
						[](const std::string &str) -> long
						{
							return std::stol(str);
						})
					);
	};

	template <typename DataTypeWrapperInstantiated, typename T>
	void numIDsWrapCaller(std::function<T(const std::string &)> conv_func)
	{
		try
		{
			DataTypeWrapperInstantiated pst_tester(input_file, rand_seed,
													conv_func(val_range_strings[0]),
													conv_func(val_range_strings[1]),
													conv_func(val_range_strings[2]),
													conv_func(val_range_strings[3]),
													conv_func(search_val_string));

			numIDsWrap(pst_tester);
		}
		catch (std::invalid_argument const &ex)
		{
			std::cerr << "Invalid argument for interval value range and/or search range\n";
			std::exit(ExitStatusCodes::INVALID_ARG_ERR);
		}
	};

	// Instantiate next IPSTester type with respect to number of IDs
	template <typename IPSTesterDataTypesInstantiated>
	void numIDsWrap(IPSTesterDataTypesInstantiated ips_tester)
	{
		if (pts_with_ids)
		{
			typename IPSTesterDataTypesInstantiated::NumIDsWrapper<PointStruct, 1> ips_tester_num_ids_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 1 wrapper\n";
#endif
			
			IDTypeWrap(ips_tester_num_ids_instan);
		}
		else	// !pts_with_ids; can skip IDTypeWrap()
		{
			typename IPSTesterDataTypesInstantiated::NumIDsWrapper<PointStruct, 0> ips_tester_num_ids_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 0 wrapper\n";
#endif

			// As no ID distribution is used anyway, just place a dummy template template parameter taking one type parameter
			typename IPSTesterDataTypesInstantiated
						::NumIDsWrapper<PointStruct, 0>
						::IDTypeWrapper<std::uniform_real_distribution, void>
							ips_tester_fully_instan(ips_tester_num_ids_instan);

			testWrap(ips_tester_fully_instan);
		}
	};

	// Instantiate next IPSTester type with respect to ID type
	template <class IPSTesterDataTypeNumIDsInstantiated>
	void IDTypeWrap(IPSTesterDataTypeNumIDsInstantiated ips_tester)
	{
		if (id_type == DataType::CHAR)
		{
			// typename necessary, as compiler defaults to treating nested names as variables
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, char> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated IDType = char wrapper\n";
#endif

			if (report_IDs)
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, char>::RetTypeWrapper<char> ips_tester_id_ret_types_instan(ips_tester_id_instan);

				testWrap(ips_tester_id_ret_types_instan);
			}
			else
			{
				// Must have <> to use default template parameter
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, char>::RetTypeWrapper<> ips_tester_id_ret_types_instan(ips_tester_id_instan);
				testWrap(ips_tester_id_ret_types_instan);
			}
		}
		else if (id_type == DataType::DOUBLE)
		{
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, double> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated IDType = double wrapper\n";
#endif

			if (report_IDs)
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, double>::RetTypeWrapper<double> ips_tester_id_ret_types_instan(ips_tester_id_instan);

				testWrap(ips_tester_id_ret_types_instan);
			}
			else
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, double>::RetTypeWrapper<> ips_tester_id_ret_types_instan(ips_tester_id_instan);
				testWrap(ips_tester_id_ret_types_instan);
			}
		}
		else if (id_type == DataType::FLOAT)
		{
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, float> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated IDType = float wrapper\n";
#endif

			if (report_IDs)
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, float>::RetTypeWrapper<float> ips_tester_id_ret_types_instan(ips_tester_id_instan);

				testWrap(ips_tester_id_ret_types_instan);
			}
			else
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_real_distribution, float>::RetTypeWrapper<> ips_tester_id_ret_types_instan(ips_tester_id_instan);
				testWrap(ips_tester_id_ret_types_instan);
			}
		}
		else if (id_type == DataType::INT)
		{
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, int> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated IDType = int wrapper\n";
#endif

			if (report_IDs)
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, int>::RetTypeWrapper<int> ips_tester_id_ret_types_instan(ips_tester_id_instan);

				testWrap(ips_tester_id_ret_types_instan);
			}
			else
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, int>::RetTypeWrapper<> ips_tester_id_ret_types_instan(ips_tester_id_instan);
				testWrap(ips_tester_id_ret_types_instan);
			}
		}
		else if (id_type == DataType::LONG)
		{
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, long> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated IDType = long wrapper\n";
#endif

			if (report_IDs)
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, long>::RetTypeWrapper<long> ips_tester_id_ret_types_instan(ips_tester_id_instan);

				testWrap(ips_tester_id_ret_types_instan);
			}
			else
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, long>::RetTypeWrapper<> ips_tester_id_ret_types_instan(ips_tester_id_instan);
				testWrap(ips_tester_id_ret_types_instan);
			}
		}
		else if (id_type == DataType::UNSIGNED_INT)
		{
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, unsigned> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated IDType = unsigned int wrapper\n";
#endif

			if (report_IDs)
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, unsigned>::RetTypeWrapper<unsigned> ips_tester_id_ret_types_instan(ips_tester_id_instan);

				testWrap(ips_tester_id_ret_types_instan);
			}
			else
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, unsigned>::RetTypeWrapper<> ips_tester_id_ret_types_instan(ips_tester_id_instan);
				testWrap(ips_tester_id_ret_types_instan);
			}
		}
		else if (id_type == DataType::UNSIGNED_LONG)
		{
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, unsigned long> ips_tester_id_instan(ips_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated IDType = unsigned long wrapper\n";
#endif

			if (report_IDs)
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, unsigned long>::RetTypeWrapper<unsigned long> ips_tester_id_ret_types_instan(ips_tester_id_instan);

				testWrap(ips_tester_id_ret_types_instan);
			}
			else
			{
				typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<std::uniform_int_distribution, unsigned long>::RetTypeWrapper<> ips_tester_id_ret_types_instan(ips_tester_id_instan);
				testWrap(ips_tester_id_ret_types_instan);
			}
		}
	};

	// Run test
	template <typename IPSTesterClass>
	void testWrap(IPSTesterClass ips_tester)
	{
#ifdef DEBUG_WRAP
		std::cout << "Beginning test\n";
#endif

		ips_tester(num_elems, num_thread_blocks, threads_per_block);
	};
};


#endif
