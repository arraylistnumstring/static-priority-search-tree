#include <string>	// To use stoi() and string operators for command-line argument parsing

#include "exit-status-codes.h"		// For consistent exit status codes
#include "pst-test-info-struct.h"

int main(int argc, char *argv[])
{
#ifdef DEBUG_TEST
	std::cout << "Began PST tester driver\n";
#endif
	PSTTestInfoStruct test_info;

	// Parse command-line arguments
	for (int i = 1; i < argc; i++)
	{
		std::string arg(argv[i]);	// Allow use of string's operators and functions

		// Help message
		if (arg == "-h" || arg == "--help")
		{
			std::cerr << "Usage: ./pst-rand-data-tester-driver <datatype-flag> <test-type-flag> ";
			std::cerr << "<tree-type-flag> ";
			std::cerr << "[-I ID_TYPE] ";
			std::cerr << "[-O] ";
			std::cerr << "[-r] ";
			std::cerr << "[-S RAND_SEED] ";
			std::cerr << "[-t] ";
			std::cerr << "[-w WARPS_PER_BLOCK] ";
			std::cerr << "-b MIN_VAL MAX_VAL [SIZE_BOUND_1] [SIZE_BOUND_2] ";
			std::cerr << "-n NUM_ELEMS";
			std::cerr << "\n\n";

			std::cerr << "\tdatatype-flag:\n";
			std::cerr << "\t\t-d, --double\tUse doubles as values\n\n";
			std::cerr << "\t\t-f, --float\tUse floats as values\n\n";
			std::cerr << "\t\t-i, --int\tUse ints for values\n\n";
			std::cerr << "\t\t-l, --long\tUse longs as values\n\n";

			std::cerr << "\ttest-type-flag:\n";
			std::cerr << "\t\t-C, --construct\n";
			std::cerr << "\t\t\t\tConstruction-only test\n\n";
			std::cerr << "\t\t-L, --left MAX_DIM1_VAL MIN_DIM2_VAL\n";
			std::cerr << "\t\t\t\tLeftwards search, treating MAX_DIM1_VAL as maximum dimension-1 value and MIN_DIM2_VAL as minimum dimension-2 value\n\n";
			std::cerr << "\t\t-R, --right MIN_DIM1_VAL MIN_DIM2_VAL\n";
			std::cerr << "\t\t\t\tRightwards search, treating MIN_DIM1_VAL as minimum dimension-1 value and MIN_DIM2_VAL as minimum dimension-2 value\n\n";
			std::cerr << "\t\t-T, --three MIN_DIM1_VAL MAX_DIM1_VAL MIN_DIM2_VAL\n";
			std::cerr << "\t\t\t\tThree-sided search, treating MIN_DIM1_VAL as minimum dimension-1 value, MAX_DIM1_VAL as maximum dimension-1 value and MIN_DIM2_VAL as minimum dimension-2 value\n\n";
			std::cerr << "\tAll search bounds are inclusive\n\n";

			std::cerr << "\ttree-type-flag:\n";
			std::cerr << "\t\t-g, --gpu\tUse StaticPSTGPU\n\n";
			std::cerr << "\t\t--iter\t\tUse StaticPSTCPUIter\n\n";
			std::cerr << "\t\t--recur\t\tUse StaticPSTCPURecur\n\n";

			std::cerr << "\t-b, --val-bounds MIN_VAL MAX_VAL [SIZE_BOUND_1] [SIZE_BOUND_2]\n";
			std::cerr << "\t\t\t\tBounds of values (inclusive) to use when generating random values for PST; must be castable to chosen datatype; when non-negative values SIZE_BOUND_1 and SIZE_BOUND_2 are specified, the lower bound of the interval is drawn from the range [MIN_VAL, MAX_VAL], and the upper bound is equal to the lower bound plus a value drawn from the range [SIZE_BOUND_1, SIZE_BOUND_2]; when only SIZE_BOUND_1 is specified, the added value is drawn from the range [0, SIZE_BOUND_1]\n\n";

			std::cerr << "\t-I, --ids ID_TYPE\tToggles assignment of IDs to the nodes of the tree with data type ID_TYPE; defaults to false; valid arguments for ID_TYPE are char, double, float, int, long, unsigned (equivalent to unsigned-int), unsigned-int, unsigned-long\n\n";

			std::cerr << "\t-n, --num-elems NUM_ELEMS\n";
			std::cerr << "\t\t\t\tNumber of elements to put in tree\n\n";

			std::cerr << "\t-O, --ordered-vals\tWhether to order values such that dimension-1 values are always less than or equal to their paired dimension-2 values; defaults to false\n\n";

			std::cerr << "\t-r, --report-IDs\tWhether to report point IDs or full info of a point; defaults to full info; if no ID type is specified, always reports full info\n\n";

			std::cerr << "\t-S, --rand-seed RAND_SEED\n";
			std::cerr << "\t\t\t\tRandom seed to use when generating data for tree; defaults to 0\n\n";

			std::cerr << "\t-t, --timed\t\tToggles timing of the construction and search portion of the code; uses on-device functions for GPU PST; defaults to false\n\n";

			std::cerr << "\t-w, --warps-per-block WARPS_PER_BLOCK\n";
			std::cerr << "\t\t\t\tNumber of warps to use in a CUDA thread block; only relevant when -g option is invoked; defaults to 1\n\n";

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

		// Test-type parsing
		else if (arg == "-C" || arg == "--construct")
			test_info.test_type = PSTTestCodes::CONSTRUCT;
		else if (arg == "-L" || arg == "--left"
					|| arg == "-R" || arg == "--right"
					|| arg == "-T" || arg == "--three"
				)
		{
			PSTTestInfoStruct::NumSearchVals num_search_vals;

			if (arg == "-L" || arg == "--left")
			{
				num_search_vals = PSTTestInfoStruct::NumSearchVals::NUM_VALS_TWO_SEARCH;
				test_info.test_type = PSTTestCodes::LEFT_SEARCH;
			}
			else if (arg == "-R" || arg == "--right")
			{
				num_search_vals = PSTTestInfoStruct::NumSearchVals::NUM_VALS_TWO_SEARCH;
				test_info.test_type = PSTTestCodes::RIGHT_SEARCH;
			}
			else	// arg == "-T" || arg == "--three"
			{
				num_search_vals = PSTTestInfoStruct::NumSearchVals::NUM_VALS_THREE_SEARCH;
				test_info.test_type = PSTTestCodes::THREE_SEARCH;
			}

			// Consume requisite number of arguments for later conversion to search range values
			for (int j = 0; j < num_search_vals; j++)
			{
				i++;
				if (i >= argc)
				{
					std::cerr << "Insufficient number of arguments provided for search bounds\n";
					return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
				}

				test_info.search_range_strings[j] = argv[i];
			}

			// For non-three-sided searches, move dimension-2 value to third slot of test_info.search_range_strings
			if (num_search_vals == PSTTestInfoStruct::NumSearchVals::NUM_VALS_TWO_SEARCH)
				test_info.search_range_strings[num_search_vals]
					= test_info.search_range_strings[num_search_vals - 1];
		}

		// Tree type parsing
		else if (arg == "-g" || arg == "--gpu")
			test_info.tree_type = PSTType::GPU;
		else if (arg == "--iter")
			test_info.tree_type = PSTType::CPU_ITER;
		else if (arg == "--recur")
			test_info.tree_type = PSTType::CPU_RECUR;

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

			if (id_type_string == "char")
				test_info.id_type = DataType::CHAR;
			else if (id_type_string == "double")
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

		// Ordered values flag parsing
		else if (arg == "-O" || arg == "--ordered-vals")
			test_info.ordered_vals = true;

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
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for random seed: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}

		// CUDA code timing flag
		else if (arg == "-t" || arg == "--timed")
			test_info.timed = true;

		// Warps per block parsing
		else if (arg == "-w" || arg == "--warps-per-block")
		{
			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for number of warps per thread block\n";
				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			try
			{
				test_info.warps_per_block = std::stoul(argv[i], nullptr, 0);
			}
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for number of warps per thread block: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}

		// Tree value parsing
		else if (arg == "-b" || arg == "--val-bounds")
		{
			for (int j = 0; j < PSTTestInfoStruct::MAX_NUM_VALS_INT_BOUNDS; j++)
			{
				if (j < PSTTestInfoStruct::MIN_NUM_VALS_INT_BOUNDS)
				{
					i++;
					if (i >= argc)
					{
						std::cerr << "Insufficient number of arguments provided for tree value bounds\n";
						return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
					}

					test_info.tree_val_range_strings[j] = argv[i];
				}
				// Test for optional presence of third and fourth arguments
				else	// j >= PSTTestInfoStruct::MIN_NUM_VALS_INT_BOUNDS
				{
					// If no more arguments can be parsed, or next argument is a new flag, avoid incrementing i and end loop over j
					if (i + 1 >= argc || argv[i + 1][0] == '-')
						break;
					else
					{
						i++;
						test_info.tree_val_range_strings[j] = argv[i];
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
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for number of elements: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}

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
