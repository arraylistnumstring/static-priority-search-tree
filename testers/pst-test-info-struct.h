#ifndef PST_TEST_INFO_STRUCT_H
#define PST_TEST_INFO_STRUCT_H

#include <algorithm>
#include <cstdlib>		// To use std::exit()
#include <functional>	// To use std::function
#include <random>
#include <string>

#include "exit-status-codes.h"
#include "point-struct.h"
#include "pst-tester.h"
#include "static-pst-cpu-iter.h"
#include "static-pst-cpu-recur.h"
#include "static-pst-gpu.h"
#include "preprocessor-symbols.h"


// Struct for information necessary to instantiate PSTTester
struct PSTTestInfoStruct
{
	// Ordering of fields chosen to minimise size of struct; std::string type appears to take 32 bytes

	std::string input_file = "";

	std::string pt_grid_dim_strings[NUM_DIMS] = {"0", "0", "0"};
	std::string metacell_dim_strings[NUM_DIMS] = {"4", "0", "0"};

	enum NumSearchVals
	{
		NUM_VALS_TWO_SEARCH=2,
		NUM_VALS_THREE_SEARCH=3
	};
	std::string search_range_strings[NumSearchVals::NUM_VALS_THREE_SEARCH] = {"0", "0", "0"};

	// Number of values necessary to define the bounds of an interval
	const static size_t MIN_NUM_VALS_INT_BOUNDS = 2;
	const static size_t MAX_NUM_VALS_INT_BOUNDS = 4;
	std::string tree_val_range_strings[MAX_NUM_VALS_INT_BOUNDS] = {"0", "0", "-1", "-1"};

	size_t rand_seed = 0;
	size_t num_elems;

	// Data types chosen to correspond to CUDA data types for correpsonding on-device values; as number of thread blocks and threads per block are both unsigned values, warps_per_block (which has value at most equal to threads per block) has been chosen similarly
	unsigned warps_per_block = 1;

	// By the standard, enums must be capable of holding int values, though the actual data-type can be char, signed int or unsigned int, as long as the chosen type can hold all values in the enumeration
	DataType data_type;
	PSTTestCodes test_type;
	PSTType tree_type;
	DataType id_type;

	// Only queried when generating random data, but set consistently with the operation being done in all cases for ease of debugging
	bool ordered_vals = false;
	bool pts_with_ids = false;
	bool report_IDs = false;
	bool timed = false;

	// Instantiate outermost PSTTester type with respect to CUDA timing
	void test()
	{
		// Must explicitly set class instantiation variables to true and false in each branch in order for code to be compile-time determinable
		if (timed)
			dataTypeWrap<PSTTester<true>>();
		else
			dataTypeWrap<PSTTester<false>>();
	};

	template <class PSTTesterTimingDet>
	void dataTypeWrap()
	{
		if (data_type == DataType::DOUBLE)
		{
			// typename necessary, as compiler defaults to treating nested names as variables
			// Function pointers cannot have default arguments (and presumably std::functions operate the same way, at least when passed as parameters), so wrap std::stod in a lambda function with the correct signature
			// Casting necessary as compiler fails to recognise a lambda as an std::function of the same signature and return type, thereby failing to instantiate template function
			treeTypeWrapCaller<typename PSTTesterTimingDet::DataTypeWrapper<double, std::uniform_real_distribution>>(static_cast<std::function<double(const std::string &)>>(
						[](const std::string &str) -> double
						{
							return std::stod(str);
						})
					);
		}
		else if (data_type == DataType::FLOAT)
		{
			treeTypeWrapCaller<typename PSTTesterTimingDet::DataTypeWrapper<float, std::uniform_real_distribution>>(static_cast<std::function<float(const std::string &)>>(
						[](const std::string &str) -> float
						{
							return std::stof(str);
						})
					);
		}
		else if (data_type == DataType::INT)
		{
			treeTypeWrapCaller<typename PSTTesterTimingDet::DataTypeWrapper<int, std::uniform_int_distribution>>(static_cast<std::function<int(const std::string &)>>(
					[](const std::string &str) -> int
					{
						return std::stoi(str);
					})
				);
		}
		else if (data_type == DataType::LONG)
		{
			treeTypeWrapCaller<typename PSTTesterTimingDet::DataTypeWrapper<long, std::uniform_int_distribution>>(static_cast<std::function<long(const std::string &)>>(
						[](const std::string &str) -> long
						{
							return std::stol(str);
						})
					);
		}
	};

	template <typename DataTypeWrapperInstantiated, typename T>
	void treeTypeWrapCaller(std::function<T(const std::string &)> conv_func)
	{
		try
		{
			DataTypeWrapperInstantiated pst_tester(input_file, rand_seed,
													conv_func(tree_val_range_strings[0]),
													conv_func(tree_val_range_strings[1]),
													conv_func(tree_val_range_strings[2]),
													conv_func(tree_val_range_strings[3]),
													conv_func(search_range_strings[0]),
													conv_func(search_range_strings[1]),
													conv_func(search_range_strings[2]),
													ordered_vals);

			treeTypeWrap(pst_tester);
		}
		catch (std::invalid_argument const &ex)
		{
			std::cerr << "Invalid argument for tree value range and/or search range\n";
			std::exit(ExitStatusCodes::INVALID_ARG_ERR);
		}
	}

	// Instantiate next PSTTester type with respect to tree type
	template <typename PSTTesterDataTypeInstantiated>
	void treeTypeWrap(PSTTesterDataTypeInstantiated pst_tester)
	{
		if (tree_type == PSTType::CPU_ITER)
		{
			typename PSTTesterDataTypeInstantiated::TreeTypeWrapper<PointStruct, StaticPSTCPUIter, PSTType::CPU_ITER> pst_tester_tree_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated StaticPSTCPUIter wrapper\n";
#endif

			numIDsWrap(pst_tester_tree_instan);
		}
		else if (tree_type == PSTType::CPU_RECUR)
		{
			typename PSTTesterDataTypeInstantiated::TreeTypeWrapper<PointStruct, StaticPSTCPURecur, PSTType::CPU_RECUR> pst_tester_tree_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated StaticPSTCPURecur wrapper\n";
#endif

			numIDsWrap(pst_tester_tree_instan);
		}
		else	// tree_type == PSTType::GPU
		{
			typename PSTTesterDataTypeInstantiated::TreeTypeWrapper<PointStruct, StaticPSTGPU, PSTType::GPU> pst_tester_tree_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated StaticPSTGPU wrapper\n";
#endif

			numIDsWrap(pst_tester_tree_instan);
		}
	};

	// Instantiate next PSTTester type with respect to number of IDs
	template <typename PSTTesterDataTreeTypesInstantiated>
	void numIDsWrap(PSTTesterDataTreeTypesInstantiated pst_tester)
	{
		if (pts_with_ids)
		{
			typename PSTTesterDataTreeTypesInstantiated::NumIDsWrapper<1> pst_tester_num_ids_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 1 wrapper\n";
#endif
			
			IDTypeWrap(pst_tester_num_ids_instan);
		}
		else	// !pts_with_ids; can skip IDTypeWrap()
		{
			typename PSTTesterDataTreeTypesInstantiated::NumIDsWrapper<0> pst_tester_num_ids_instan(pst_tester);

#ifdef DEBUG_WRAP
			std::cout << "Instantiated num_IDs = 0 wrapper\n";
#endif

			// As no ID distribution is used anyway, just place a dummy template template parameter taking one type parameter
			typename PSTTesterDataTreeTypesInstantiated
						::NumIDsWrapper<0>
						::IDTypeWrapper<std::uniform_real_distribution, void>
							pst_tester_fully_instan(pst_tester_num_ids_instan);

			testWrap(pst_tester_fully_instan);
		}
	};

	// Instantiate next PSTTester type with respect to ID type for randomly generated IDs
	template <class PSTTesterDataTreeTypesNumIDsInstantiated>
	void IDTypeWrap(PSTTesterDataTreeTypesNumIDsInstantiated pst_tester)
	{
		if (id_type == DataType::CHAR)
			IDTypeWrapInstantiator<std::uniform_int_distribution, char>(pst_tester,
										static_cast<std::function<char(const std::string &)>>(
													[](const std::string &str) -> char
													{
														return static_cast<char>(std::stoi(str));
													})
									);
		else if (id_type == DataType::DOUBLE)
			IDTypeWrapInstantiator<std::uniform_real_distribution, double>(pst_tester,
										static_cast<std::function<double(const std::string &)>>(
													[](const std::string &str) -> double
													{
														return std::stod(str);
													})
									);
		else if (id_type == DataType::FLOAT)
			IDTypeWrapInstantiator<std::uniform_real_distribution, float>(pst_tester,
										static_cast<std::function<float(const std::string &)>>(
													[](const std::string &str) -> float
													{
														return std::stof(str);
													})
									);
		else if (id_type == DataType::INT)
			IDTypeWrapInstantiator<std::uniform_int_distribution, int>(pst_tester,
										static_cast<std::function<int(const std::string &)>>(
													[](const std::string &str) -> int
													{
														return std::stoi(str);
													})
									);
		else if (id_type == DataType::LONG)
			IDTypeWrapInstantiator<std::uniform_int_distribution, long>(pst_tester,
										static_cast<std::function<long(const std::string &)>>(
													[](const std::string &str) -> long
													{
														return std::stol(str);
													})
									);
		else if (id_type == DataType::UNSIGNED_INT)
			IDTypeWrapInstantiator<std::uniform_int_distribution, unsigned>(pst_tester,
										static_cast<std::function<unsigned(const std::string &)>>(
													[](const std::string &str) -> unsigned
													{
														return static_cast<unsigned>(std::stoul(str));
													})
									);
		else if (id_type == DataType::UNSIGNED_LONG)
			IDTypeWrapInstantiator<std::uniform_int_distribution, unsigned long>(pst_tester,
										static_cast<std::function<unsigned long(const std::string &)>>(
													[](const std::string &str) -> unsigned long
													{
														return std::stoul(str);
													})
									);
	};

	template <template<typename> typename IDDistr, typename IDType, typename PSTTesterDataTreeTypesNumIDsInstantiated>
	void IDTypeWrapInstantiator(PSTTesterDataTreeTypesNumIDsInstantiated pst_tester, std::function<IDType(const std::string &)> conv_func)
	{
		if (input_file == "")	// Randomly-generated IDs
		{
			typename PSTTesterDataTreeTypesNumIDsInstantiated::IDTypeWrapper<IDDistr, IDType> pst_tester_id_instan(pst_tester);

			reportIDsWrap<IDType>(pst_tester_id_instan);
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
				{
					metacell_dims[i] = metacell_dims[0];
				}
			}

			typename PSTTesterDataTreeTypesNumIDsInstantiated::IDTypeWrapper<IDDistr, IDType> pst_tester_id_instan(pst_tester, pt_grid_dims, metacell_dims);

			reportIDsWrap<IDType>(pst_tester_id_instan);
		}
	}

	template <typename IDType, typename PSTTesterDataTreeIDTypesInstantiated>
	void reportIDsWrap(PSTTesterDataTreeIDTypesInstantiated pst_tester)
	{
		if (report_IDs)
		{
			typename PSTTesterDataTreeIDTypesInstantiated::RetTypeWrapper<IDType> pst_tester_fully_instan(pst_tester);

			testWrap(pst_tester_fully_instan);
		}
		else
		{
			typename PSTTesterDataTreeIDTypesInstantiated::RetTypeWrapper<> pst_tester_fully_instan(pst_tester);

			testWrap(pst_tester_fully_instan);
		}
	};

	// Run test
	template <typename PSTTesterClass>
	void testWrap(PSTTesterClass pst_tester)
	{
#ifdef DEBUG_WRAP
		std::cout << "Beginning test\n";
#endif

		pst_tester(num_elems, warps_per_block, test_type);
	};
};


#endif
