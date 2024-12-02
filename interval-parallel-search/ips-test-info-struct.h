#ifndef IPS_TEST_INFO_STRUCT_H
#define IPS_TEST_INFO_STRUCT_H

#include <algorithm>
#include <cstdlib>		// To use std::exit()
#include <random>
#include <string>

#include "exit-status-codes.h"
#include "ips-tester.h"
#include "point-struct.h"
#include "preprocessor-symbols.h"


// Struct for information necessary to instantiate IPSTester
struct IPSTestInfoStruct
{
	// Ordering of fields chosen to minimise size of struct; std::string type appears to take 32 bytes

	std::string input_file = "";

	std::string pt_grid_dim_strings[NUM_DIMS] = {"0", "0", "0"};
	std::string metacell_dim_strings[NUM_DIMS] = {"4", "0", "0"};

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
#ifdef DEBUG_WRAP
		std::cout << "Began dataTypeWrap()\n";
		std::cout << "Table of data type values:\n";
		std::cout << "Double: " << DataType::DOUBLE << '\n';
		std::cout << "Float: " << DataType::FLOAT << '\n';
		std::cout << "Int: " << DataType::INT << '\n';
		std::cout << "Long: " << DataType::LONG << '\n';
		std::cout << '\n';
		std::cout << "Data type is " << data_type << '\n';
#endif
		if (data_type == DataType::DOUBLE)
			// typename necessary, as compiler defaults to treating nested names as variables
			/*
				For functions with default arguments, outside of directly calling that function by name, default arguments are not visible, so when binding functions, type signatures treat all parameters the same.
				For example, an invocation of e.g. std::stod() is treated as a signature of std::function<double(const std::string &, std::size_t *)> (even though the second argument has a default value), and thus cannot be assigned to a function with signature std::function<double(const std::string &)>
				Hence, in general, to solve such issues, wrap the desired function in a lambda function with the correct signature.
			*/
			// Casting necessary as compiler fails to recognise a lambda as an std::function of the same signature and return type (instead recognising it as a lambda [](const std::string &)->double), thereby failing to instantiate template function
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
#ifdef DEBUG_WRAP
		std::cout << "Began numIDsWrapCaller()\n";
#endif
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
#ifdef DEBUG_WRAP
		std::cout << "Began numIDsWrap()\n";
#endif
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
			IDTypeWrapInstantiator<std::uniform_int_distribution, char>(ips_tester,
										static_cast<std::function<char(const std::string &)>>(
													[](const std::string &str) -> char
													{
														return static_cast<char>(std::stoi(str));
													})
									);
		else if (id_type == DataType::DOUBLE)
			IDTypeWrapInstantiator<std::uniform_real_distribution, double>(ips_tester,
										static_cast<std::function<double(const std::string &)>>(
													[](const std::string &str) -> double
													{
														return std::stod(str);
													})
									);
		else if (id_type == DataType::FLOAT)
			IDTypeWrapInstantiator<std::uniform_real_distribution, float>(ips_tester,
										static_cast<std::function<float(const std::string &)>>(
													[](const std::string &str) -> float
													{
														return std::stof(str);
													})
									);
		else if (id_type == DataType::INT)
			IDTypeWrapInstantiator<std::uniform_int_distribution, int>(ips_tester,
										static_cast<std::function<int(const std::string &)>>(
													[](const std::string &str) -> int
													{
														return std::stoi(str);
													})
									);
		else if (id_type == DataType::LONG)
			IDTypeWrapInstantiator<std::uniform_int_distribution, long>(ips_tester,
										static_cast<std::function<long(const std::string &)>>(
													[](const std::string &str) -> long
													{
														return std::stol(str);
													})
									);
		else if (id_type == DataType::UNSIGNED_INT)
			IDTypeWrapInstantiator<std::uniform_int_distribution, unsigned>(ips_tester,
										static_cast<std::function<unsigned(const std::string &)>>(
													[](const std::string &str) -> unsigned
													{
														return static_cast<unsigned>(std::stoul(str));
													})
									);
		else if (id_type == DataType::UNSIGNED_LONG)
			IDTypeWrapInstantiator<std::uniform_int_distribution, unsigned long>(ips_tester,
										static_cast<std::function<unsigned long(const std::string &)>>(
													[](const std::string &str) -> unsigned long
													{
														return std::stoul(str);
													})
									);
	};

	template <template<typename> typename IDDistr, typename IDType, typename IPSTesterDataTypeNumIDsInstantiated>
	void IDTypeWrapInstantiator(IPSTesterDataTypeNumIDsInstantiated ips_tester, std::function<IDType(const std::string &)> conv_func)
	{
		if (input_file == "")	// Randomly-generated IDs
		{
			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<IDDistr, IDType> ips_tester_id_instan(ips_tester);

			reportIDsWrap<IDType>(ips_tester_id_instan);
		}
		else
		{
			IDType pt_grid_dims[NUM_DIMS];
			IDType metacell_dims[NUM_DIMS];
			for (int i = 0; i < NUM_DIMS; i++)
			{
				try
				{
					pt_grid_dims[i] = conv_func(pt_grid_dim_strings[i]);
					metacell_dims[i] = conv_func(metacell_dim_strings[i]);
				}
				catch (std::invalid_argument const &ex)
				{
					std::cerr << "Invalid argument for point grid and/or metacell dimension " << i << '\n';
					std::exit(ExitStatusCodes::INVALID_ARG_ERR);
				}

				if (pt_grid_dims[i] <= 0)
				{
					std::cerr << "Invalid value of " << pt_grid_dims[i] << " for point grid dimension " << i << '\n';
					std::exit(ExitStatusCodes::INVALID_ARG_ERR);
				}
				if ( (i == 0  && metacell_dims[i] == 0)
						// Check whether IDType is unsigned to silence "meaningless comparison with 0" warnings
						|| (!std::is_unsigned<IDType>::value && metacell_dims[i] < 0) )
				{
					std::cerr << "Invalid value of " << metacell_dims[i] << " for metacell dimension " << i << '\n';
					std::exit(ExitStatusCodes::INVALID_ARG_ERR);
				}
				else if (metacell_dims[i] == 0)
					metacell_dims[i] = metacell_dims[0];
			}

			typename IPSTesterDataTypeNumIDsInstantiated::IDTypeWrapper<IDDistr, IDType> ips_tester_id_instan(ips_tester, pt_grid_dims, metacell_dims);

			reportIDsWrap<IDType>(ips_tester_id_instan);
		}
	}

	template <typename IDType, typename IPSTesterDataIDTypesInstantiated>
	void reportIDsWrap(IPSTesterDataIDTypesInstantiated ips_tester)
	{
		if (report_IDs)
		{
			typename IPSTesterDataIDTypesInstantiated::RetTypeWrapper<IDType> ips_tester_fully_instan(ips_tester);

			testWrap(ips_tester_fully_instan);
		}
		else
		{
			typename IPSTesterDataIDTypesInstantiated::RetTypeWrapper<> ips_tester_fully_instan(ips_tester);

			testWrap(ips_tester_fully_instan);
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
