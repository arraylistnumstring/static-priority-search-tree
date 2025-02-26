#include <algorithm>	// To use transform()
#include <string>		// To use stoi() and string operators for command-line argument parsing

#include "exit-status-codes.h"		// For consistent exit status codes
#include "ips-tester.h"
#include "ips-test-info-struct.h"


#ifdef DEBUG_TEST
#include <iostream>
#endif


int main(int argc, char *argv[])
{
#ifdef DEBUG_TEST
	std::cout << "Began interval parallel search tester driver\n";
#endif
	IPSTestInfoStruct test_info;

	// Parse command-line arguments
	for (int i = 1; i < argc; i++)
	{
		std::string arg(argv[i]);	// Allow use of string's operators and functions
#ifdef DEBUG_TEST
		std::cout << "Command line argument detected: " << arg << '\n';
		std::cout << "Argument number: " << i << '\n';
#endif

		// Help message
		if (arg == "-h" || arg == "--help")
		{
			std::cerr << "Usage: ./ips-rand-data-tester-driver <datatype-flag> ";
			std::cerr << "[-I ID_TYPE] ";
			std::cerr << "[-r] ";
			std::cerr << "[-S RAND_SEED] ";
			std::cerr << "[-T THREADS_PER_BLOCK] ";
			std::cerr << "[-t] ";
			std::cerr << "-B NUM_BLOCKS ";
			std::cerr << "-b MIN_VAL MAX_VAL [SIZE_BOUND_1] [SIZE_BOUND_2] ";
			std::cerr << "-n NUM_ELEMS ";
			std::cerr << "-s SEARCH_VAL";
			std::cerr << "\n\n";

			std::cerr << "\tdatatype-flag:\n";
			std::cerr << "\t\t-d, --double\tUse doubles as values\n\n";
			std::cerr << "\t\t-f, --float\tUse floats as values\n\n";
			std::cerr << "\t\t-i, --int\tUse ints for values\n\n";
			std::cerr << "\t\t-l, --long\tUse longs as values\n\n";

			std::cerr << "\t-B, --num-blocks NUM_BLOCKS\n";
			std::cerr << "\t\t\t\tNumber of blocks to use in grid for CUDA kernel\n\n";

			std::cerr << "\t-b, --val-bounds MIN_VAL MAX_VAL [SIZE_BOUND_1] [SIZE_BOUND_2]\n";
			std::cerr << "\t\t\t\tBounds of values (inclusive) to use when generating random values to search on; must be castable to chosen datatype; when non-negative values SIZE_BOUND_1 and SIZE_BOUND_2 are specified, the lower bound of the interval is drawn from the range [MIN_VAL, MAX_VAL], and the upper bound is equal to the lower bound plus a value drawn from the range [SIZE_BOUND_1, SIZE_BOUND_2]; when only SIZE_BOUND_1 is specified, the added value is drawn from the range [0, SIZE_BOUND_1]\n\n";

			std::cerr << "\t-I, --ids ID_TYPE\tToggles assignment of IDs of data type ID_TYPE to input points; defaults to false; valid arguments for ID_TYPE are double, float, int, long, unsigned (equivalent to unsigned-int), unsigned-int, unsigned-long\n\n";

			std::cerr << "\t-n, --num-elems NUM_ELEMS\n";
			std::cerr << "\t\t\t\tNumber of elements to put in tree\n\n";

			std::cerr << "\t-r, --report-IDs\tWhether to report point IDs or full info of a point; defaults to full info; if no ID type is specified, always reports full info\n\n";

			std::cerr << "\t-S, --rand-seed RAND_SEED\n";
			std::cerr << "\t\t\t\tRandom seed to use when generating data for tree; defaults to 0\n\n";

			std::cerr << "\t-s, --search-val SEARCH_VAL\n";
			std::cerr << "\t\t\t\tValue to search for in all intervals; interval bounds are treated as inclusive\n\n";

			std::cerr << "\t-T, --threads-per-block THREADS_PER_BLOCK\n";
			std::cerr << "\t\t\t\tNumber of threads to use in a thread block; defaults to 128, the number of threads originally used by Liu et al. for the interval parallel search (compaction) portion of their procedure\n\n";

			std::cerr << "\t-t, --timed-CUDA\tToggles timing of the CUDA portion of the code using on-device functions; defaults to false\n\n";

			return ExitStatusCodes::SUCCESS;
		}

		// Data-type parsing
		else if (arg == "-d" || arg == "--double")
			test_info.data_type = DataType::DOUBLE;
		else if (arg == "-f" || arg == "--float")
			test_info.data_type = DataType::FLOAT;
		else if (arg == "-i" || arg == "--int")
			test_info.data_type = DataType::INT;
		else if (arg == "-l" || arg == "--long")
			test_info.data_type = DataType::LONG;

		// ID flag and ID type parsing
		else if (arg == "-I" || arg == "--ids")
		{
			test_info.pts_with_ids = true;

			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for ID data type\n";
				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			// Convert id_type_string to lowercase for easier processing
			std::string id_type_string(argv[i]);
			std::transform(id_type_string.begin(), id_type_string.end(),
							id_type_string.begin(),
							[](unsigned char c){ return std::tolower(c); });

			if (id_type_string == "double")
				test_info.id_type = DataType::DOUBLE;
			else if (id_type_string == "float")
				test_info.id_type = DataType::FLOAT;
			else if (id_type_string == "int")
				test_info.id_type = DataType::INT;
			else if (id_type_string == "long")
				test_info.id_type = DataType::LONG;
			else if (id_type_string == "unsigned" || id_type_string == "unsigned-int")
				test_info.id_type = DataType::UNSIGNED_INT;
			else if (id_type_string == "unsigned-long")
				test_info.id_type = DataType::UNSIGNED_LONG;
			else
			{
				std::cerr << "Invalid argument for ID data type: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}
		
		// Report ID flag parsing
		else if (arg == "-r" || arg == "--report-IDs")
			test_info.report_IDs = true;

		// Random seed parsing
		else if (arg == "-S" || arg == "--rand-seed")
		{
			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for random seed\n";
				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			try
			{
				// Curly braces necessary around try blocks
				test_info.rand_seed = std::stoull(argv[i], nullptr, 0);
			}
			catch (const std::invalid_argument &ex)
			{
				std::cerr << "Invalid argument for random seed: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}

		// Number of threads per block parsing
		else if (arg == "-T" || arg == "--threads-per-block")
		{
			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for number of threads per block\n";
				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			try
			{
				test_info.threads_per_block = std::stoul(argv[i]);
			}
			catch (const std::invalid_argument &ex)
			{
				std::cerr << "Invalid argument for number of threads per block: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}

		// CUDA code timing flag
		else if (arg == "-t" || arg == "--timed-CUDA")
			test_info.timed_CUDA = true;

		// Number of thread blocks parsing
		else if (arg == "-B" || arg == "--num-blocks")
		{
			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for number of thread blocks\n";

				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			try
			{
				test_info.num_thread_blocks = std::stoul(argv[i]);
			}
			catch (const std::invalid_argument &ex)
			{
				std::cerr << "Invalid argument for number of thread blocks: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}

		// Interval value bound parsing
		else if (arg == "-b" || arg == "--val-bounds")
		{
			for (int j = 0; j < IPSTestInfoStruct::MAX_NUM_VALS_INT_BOUNDS; j++)
			{
				if (j < IPSTestInfoStruct::MIN_NUM_VALS_INT_BOUNDS)
				{
					i++;
					if (i >= argc)
					{
						std::cerr << "Insufficient number of arguments provided for interval value bounds\n";
						return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
					}

					test_info.val_range_strings[j] = argv[i];
				}
				// Test for optional presence of third and fourth arguments
				else	// j >= IPSTestInfoStruct::MIN_NUM_VALS_INT_BOUNDS)
				{
					// If no more arguments can be parsed, or next argument is a new flag, avoid incrementing i and end loop over j
					if (i + 1 >= argc || argv[i + 1][0] == '-')
						break;
					else
					{
						i++;
						test_info.val_range_strings[j] = argv[i];
					}
				}
			}
		}

		// Number of elements parsing
		else if (arg == "-n" || arg == "--num-elems")
		{
			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for number of elements\n";
				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			try
			{
				test_info.num_elems = std::stoull(argv[i], nullptr, 0);
			}
			catch (const std::invalid_argument &ex)
			{
				std::cerr << "Invalid argument for number of elements: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}

		}
	
		// Search value parsing
		else if (arg == "-s" || arg == "--search-val")
		{
			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for search value\n";
				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			test_info.search_val_string = argv[i];
		}

		else	// Invalid argument
		{
			std::cerr << "Invalid command-line argument: " << argv[i] << '\n';
			return ExitStatusCodes::INVALID_ARG_ERR;
		}
	}

#ifdef DEBUG_TEST
	std::cout << "Completed command-line argument parsing; beginning test-running code\n";
#endif

	// Run test
	test_info.test();

	return ExitStatusCodes::SUCCESS;
}
