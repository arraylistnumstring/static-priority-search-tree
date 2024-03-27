#include <string>	// To use stoi() and string operators for command-line argument parsing

#include "../point-struct.h"
#include "../static-pst-cpu-iter.h"
#include "../static-pst-cpu-recur.h"
#include "../static-pst-gpu.h"
#include "pst-tester.h"

int main(int argc, char *agrv[])
{
	enum DataType {BOOL, CHAR, DOUBLE, FLOAT, INT, LONG};

	enum NumSearchVals
	{
		NUM_VALS_TWO_SEARCH=2,
		NUM_VALS_THREE_SEARCH=3
	};

	enum PSTType {CPU_ITER, CPU_RECUR, GPU};


	DataType data_type;
	
	PSTTester::TestCodes test_type;
	
	PSTType tree_type;
	
	std::string search_range_strings[NUM_VALS_THREE_SEARCH];

	unsigned char num_ids = 0;
	DataType id_type;

	size_t rand_seed = 0;
	
	// Number of values necessary to define the bounds of an interval
	const size_t NUM_VALS_INT_BOUNDS = 2;
	std::string tree_val_range_strings[NUM_VALS_INT_BOUNDS];

	size_t num_elems;

	// Parse command-line arguments
	for (int i = 0; i < argc; i++)
	{
		std::string arg(argv[i]);	// Allow use of string's operators and functions

		// Help message
		if (arg == "-h" || arg == "--help")
		{
			std::cerr << "Usage: ./pst-tester-driver <datatype-flag> <test-type-flag> ";
			std::cerr << "<tree-type-flag> ";
			std::cerr << "[-I] "
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

			std::cerr << "\t-I, --ids-on\tToggles on IDs for nodes of the tree; defaults to false\n\n";

			std::cerr << "\t-b, --val-bounds MIN_VAL MAX_VAL\tBounds of values (inclusive) to use when generating random values for PST; must be castable to chosen datatype\n\n";

			std::cerr << "\t-n, --num-elems NUM_ELEMS\tNumber of elements to put in tree\n\n";

			std::cerr << "\t-r, --rand-seed RAND_SEED\tRandom seed to use when generating data for tree; defaults to 0\n\n";

			return 1;
		}

		// Data-type parsing
		else if (arg == "-d" || arg == "--double")
			data_type = DataType::DOUBLE;
		else if (arg == "-f" || arg == "--float")
			data_type = DataType::FLOAT;
		else if (arg == "-i" || arg == "--int")
			data_type = DataType::INT;
		else if (arg == "l" || arg == "--long")
			data_type = DataType::LONG;

		// Test-type parsing
		else if (arg == "-C" || arg == "--construct")
			test_type = PSTTester::TestCodes::CONSTRUCT;
		else if (arg == "-L" || arg == "--left"
					|| arg == "-R" || arg == "--right"
					|| arg == "-T" || arg == "--three"
				)
		{
			NumSearchVals num_search_vals;

			if (arg == "-L" || arg == "--left")
			{
				num_search_vals = NUM_VALS_TWO_SEARCH;
				test_type = PSTTester::TestCodes::LEFT_SEARCH;
			}
			else if (arg == "-R" || arg == "--right")
			{
				num_search_vals = NUM_VALS_TWO_SEARCH;
				test_type = PSTTester::TestCodes::RIGHT_SEARCH;
			}
			else	// arg == "-T" || arg == "--three"
			{
				num_search_vals = NUM_VALS_THREE_SEARCH;
				test_type = PSTTester::TestCodes::THREE_SEARCH;
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
					search_range_strings[j](argv[i]);
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
			tree_type = PSTType::GPU;
		else if (arg == "--iter")
			tree_type = PSTType::CPU_ITER;
		else if (arg == "--recur")
			tree_type = PSTType::CPU_RECUR;

		// ID flag and ID type parsing
		else if (arg == "-I" || arg == "--ids-on")
		{
			num_ids = 1;

			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for ID data type\n";
				return 2;
			}
			try
			{
				std::string id_type_string(argv[i]);
				std::transform(data.begin(), data.end(), data.begin(),
								[](unsigned char c){ return std::tolower(c); });

				if (id_type_string == "bool")
					data_type = DataType::BOOL;
				else if (id_type_string == "char")
					data_type = DataType::CHAR;
				else if (id_type_string == "double")
					data_type = DataType::DOUBLE;
				else if (id_type_string == "--float")
					data_type = DataType::FLOAT;
				else if (id_type_string == "--int")
					data_type = DataType::INT;
				else if (id_type_string == "--long")
					data_type = DataType::LONG;
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
				rand_seed = std::stoull(argv[i], nullptr, 0);
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
			for (int j = 0; j < NUM_VALS_INT_BOUNDS; j++)
			{
				i++;
				if (i >= argc)
				{
					std::cerr << "Insufficient number of arguments provided for tree value bounds\n";
					return 2;
				}
				try
				{
					tree_val_range_strings[j](argv[i]);
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
				num_elems = std::stoull(argv[i], nullptr, 0);
			}
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for number of elements: " << argv[i] << '\n';
				return 3;
			}

		}
	}

	// Instantiate PSTTester based on datatype
	if (data_type == DOUBLE)
	{
		PSTTester<double> pst_tester(rand_seed, std::stod(tree_val_range_strings[0]),
										std::stod(tree_val_range_strings[1]));

		if (tree_type == PSTType::CPU_ITER)
			pst_tester.testPST<PointStruct,
								StaticPSTCPUIter,
								id_type,
								num_ids>
									(num_elems, test_type, std::stod(search_range_strings[0]),
									 std::stod(search_range_strings[1]),
									 std::stod(search_range_strings[2])
									);
		else if (tree_type == PSTType::CPU_RECUR)
			pst_tester.testPST<PointStruct,
								StaticPSTCPURecur,
								id_type,
								num_ids>
									(num_elems, test_type, std::stod(search_range_strings[0]),
									 std::stod(search_range_strings[1]),
									 std::stod(search_range_strings[2])
									);
		else if (tree_type == PSTType::GPU)
			pst_tester.testPST<PointStruct,
								StaticPSTGPU,
								id_type,
								num_ids>
									(num_elems, test_type, std::stod(search_range_strings[0]),
									 std::stod(search_range_strings[1]),
									 std::stod(search_range_strings[2])
									);
	}
	else if (data_type == FLOAT)
	{
		PSTTester<float> pst_tester(rand_seed, std::stof(tree_val_range_strings[0]),
										std::stof(tree_val_range_strings[1]));

		if (tree_type == PSTType::CPU_ITER)
			pst_tester.testPST<PointStruct,
								StaticPSTCPUIter,
								id_type,
								num_ids>
									(num_elems, test_type, std::stof(search_range_strings[0]),
									 std::stof(search_range_strings[1]),
									 std::stof(search_range_strings[2])
									);
		else if (tree_type == PSTType::CPU_RECUR)
			pst_tester.testPST<PointStruct,
								StaticPSTCPURecur,
								id_type,
								num_ids>
									(num_elems, test_type, std::stof(search_range_strings[0]),
									 std::stof(search_range_strings[1]),
									 std::stof(search_range_strings[2])
									);
		else if (tree_type == PSTType::GPU)
			pst_tester.testPST<PointStruct,
								StaticPSTGPU,
								id_type,
								num_ids>
									(num_elems, test_type, std::stof(search_range_strings[0]),
									 std::stof(search_range_strings[1]),
									 std::stof(search_range_strings[2])
									);
	}
	else if (data_type == INT)
	{
		PSTTester<int> pst_tester(rand_seed, std::stoi(tree_val_range_strings[0]),
										std::stoi(tree_val_range_strings[1]));

		if (tree_type == PSTType::CPU_ITER)
			pst_tester.testPST<PointStruct,
								StaticPSTCPUIter,
								id_type,
								num_ids>
									(num_elems, test_type, std::stoi(search_range_strings[0]),
									 std::stoi(search_range_strings[1]),
									 std::stoi(search_range_strings[2])
									);
		else if (tree_type == PSTType::CPU_RECUR)
			pst_tester.testPST<PointStruct,
								StaticPSTCPURecur,
								id_type,
								num_ids>
									(num_elems, test_type, std::stoi(search_range_strings[0]),
									 std::stoi(search_range_strings[1]),
									 std::stoi(search_range_strings[2])
									);
		else if (tree_type == PSTType::GPU)
			pst_tester.testPST<PointStruct,
								StaticPSTGPU,
								id_type,
								num_ids>
									(num_elems, test_type, std::stoi(search_range_strings[0]),
									 std::stoi(search_range_strings[1]),
									 std::stoi(search_range_strings[2])
									);
	}
	else if (data_type == LONG)
	{
		PSTTester<long> pst_tester(rand_seed, std::stol(tree_val_range_strings[0]),
										std::stol(tree_val_range_strings[1]));

		if (tree_type == PSTType::CPU_ITER)
			pst_tester.testPST<PointStruct,
								StaticPSTCPUIter,
								id_type,
								num_ids>
									(num_elems, test_type, std::stol(search_range_strings[0]),
									 std::stol(search_range_strings[1]),
									 std::stol(search_range_strings[2])
									);
		else if (tree_type == PSTType::CPU_RECUR)
			pst_tester.testPST<PointStruct,
								StaticPSTCPURecur,
								id_type,
								num_ids>
									(num_elems, test_type, std::stol(search_range_strings[0]),
									 std::stol(search_range_strings[1]),
									 std::stol(search_range_strings[2])
									);
		else if (tree_type == PSTType::GPU)
			pst_tester.testPST<PointStruct,
								StaticPSTGPU,
								id_type,
								num_ids>
									(num_elems, test_type, std::stol(search_range_strings[0]),
									 std::stol(search_range_strings[1]),
									 std::stol(search_range_strings[2])
									);
	}

	return 0;
}
