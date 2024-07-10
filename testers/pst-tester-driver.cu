#include <random>	// To use uniform_int_distribution and uniform_real_distribution
#include <string>	// To use stoi() and string operators for command-line argument parsing

#include "pst-test-info-struct.h"

int main(int argc, char *argv[])
{
#ifdef DEBUG_TEST
	std::cout << "Began PST tester driver\n";
#endif
	PSTTestInfoStruct test_info;

	// Parse command-line arguments
	for (int i = 0; i < argc; i++)
	{
		std::string arg(argv[i]);	// Allow use of string's operators and functions

		// Help message
		if (arg == "-h" || arg == "--help")
		{
			std::cerr << "Usage: ./pst-tester-driver <datatype-flag> <test-type-flag> ";
			std::cerr << "<tree-type-flag> ";
			std::cerr << "[-I ID_TYPE] ";
			std::cerr << "[-r RAND_SEED] ";
			std::cerr << "-b MIN_VAL MAX_VAL ";
			std::cerr << "-n NUM_ELEMS";
			std::cerr << "\n\n";

			std::cerr << "\tdatatype-flag:\n";
			std::cerr << "\t\t-d, --double\tUse doubles as values\n\n";
			std::cerr << "\t\t-f, --float\tUse floats as values\n\n";
			std::cerr << "\t\t-i, --int\tUse ints for values\n\n";
			std::cerr << "\t\t-l, --long\tUse longs as values\n\n";

			std::cerr << "\ttest-type-flag:\n";
			std::cerr << "\t\t-C, --construct\tConstruction-only test\n";
			std::cerr << "\t\t-L, --left MAX_DIM1_VAL MIN_DIM2_VAL\tLeftwards search, treating MAX_DIM1_VAL as maximum dimension-1 value and MIN_DIM2_VAL as minimum dimension-2 value\n";
			std::cerr << "\t\t-R, --right MIN_DIM1_VAL MIN_DIM2_VAL\tRightwards search, treating MIN_DIM1_VAL as minimum dimension-1 value and MIN_DIM2_VAL as minimum dimension-2 value\n";
			std::cerr << "\t\t-T, --three MIN_DIM1_VAL MAX_DIM1_VAL MIN_DIM2_VAL\tThree-sided search, treating MIN_DIM1_VAL as minimum dimension-1 value, MAX_DIM1_VAL as maximum dimension-1 value and MIN_DIM2_VAL as minimum dimension-2 value\n";
			std::cerr << "\tAll search bounds are inclusive\n";
			std::cerr << '\n';

			std::cerr << "\ttree-type-flag:\n";
			std::cerr << "\t\t-g, --gpu\tUse StaticPSTGPU\n";
			std::cerr << "\t\t--iter\tUse StaticPSTCPUIter\n";
			std::cerr << "\t\t--recur\tUse StaticPSTCPURecur\n";
			std::cerr << '\n';

			std::cerr << "\t-b, --val-bounds MIN_VAL MAX_VAL\tBounds of values (inclusive) to use when generating random values for PST; must be castable to chosen datatype\n\n";

			std::cerr << "\t-I, --ids DATA_TYPE\tToggles assignment of IDs to the nodes of the tree with data type DATA_TYPE; defaults to false; valid data types are char, double, float, int, long\n\n";

			std::cerr << "\t-n, --num-elems NUM_ELEMS\tNumber of elements to put in tree\n\n";

			std::cerr << "\t-r, --rand-seed RAND_SEED\tRandom seed to use when generating data for tree; defaults to 0\n\n";

			return 1;
		}

		// Data-type parsing
		else if (arg == "-d" || arg == "--double")
			test_info.data_type = DataType::DOUBLE;
		else if (arg == "-f" || arg == "--float")
			test_info.data_type = DataType::FLOAT;
		else if (arg == "-i" || arg == "--int")
			test_info.data_type = DataType::INT;
		else if (arg == "l" || arg == "--long")
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
					return 2;
				}
				try
				{
					// Curly braces necessary around try blocks
					test_info.search_range_strings[j] = std::string(argv[i]);
				}
				catch (std::invalid_argument const &ex)
				{
					std::cerr << "Invalid argument for search range value bound: " << argv[i] << '\n';
					return 3;
				}
			}
		}

		// Tree type parsing
		else if (arg == "-g" || arg == "--gpu")
			test_info.tree_type = PSTType::GPU;
		else if (arg == "--iter")
			test_info.tree_type = PSTType::CPU_ITER;
		else if (arg == "--recur")
			test_info.tree_type = PSTType::CPU_RECUR;

		// ID flag and ID type parsing
		else if (arg == "-I" || arg == "--ids-on")
		{
			test_info.pts_with_ids = true;

			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for ID data type\n";
				return 2;
			}
			try
			{
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
			}
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for ID data type: " << argv[i] << '\n';
				return 3;
			}
		}

		// Random seed parsing
		else if (arg == "-r" || arg == "--rand-seed")
		{
			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for random seed\n";
				return 2;
			}
			try
			{
				test_info.rand_seed = std::stoull(argv[i], nullptr, 0);
			}
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for random seed: " << argv[i] << '\n';
				return 3;
			}
		}

		// Tree value parsing
		else if (arg == "-b" || arg == "--val-bounds")
		{
			for (int j = 0; j < PSTTestInfoStruct::NUM_VALS_INT_BOUNDS; j++)
			{
				i++;
				if (i >= argc)
				{
					std::cerr << "Insufficient number of arguments provided for tree value bounds\n";
					return 2;
				}
				try
				{
					test_info.tree_val_range_strings[j] = std::string(argv[i]);
				}
				catch (std::invalid_argument const &ex)
				{
					std::cerr << "Invalid argument for tree value bound: " << argv[i] << '\n';
					return 3;
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
				return 2;
			}
			try
			{
				test_info.num_elems = std::stoull(argv[i], nullptr, 0);
			}
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for number of elements: " << argv[i] << '\n';
				return 3;
			}

		}
	}

#ifdef DEBUG_TEST
	std::cout << "Completed command-line argument parsing; beginning test-running code\n";
#endif

	// Run test
	test_info.test();

	return 0;
}
