#include <filesystem>	// To use filesystem existence checks
#include <string>		// To use stoi() and string operators for command-line argument parsing

#include "exit-status-codes.h"				// For consistent exit status codes
#include "linearise-id.h"		// For NUM_DIMS definition
#include "pst-tester.h"
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
#ifdef DEBUG_TEST
		std::cout << "Command line argument detected: " << arg << '\n';
		std::cout << "Argument number: " << i << '\n';
#endif

		// Help message
		if (arg == "-h" || arg == "--help")
		{
			std::cerr << "Usage: ./pst-dataset-tester-driver data-file ";
			std::cerr << "DIM_TYPE PT_GRID_DIM_X PT_GRID_DIM_Y PT_GRID_DIM_Z ";
			std::cerr << "<datatype-flag> <tree-type-flag> ";
			std::cerr << "[-I] ";
			std::cerr << "[-m METACELL_DIM_X [METACELL_DIM_Y METACELL_DIM_Z]] ";
			std::cerr << "[-r] ";
			std::cerr << "[-t] ";
			std::cerr << "[-w WARPS_PER_BLOCK] ";
			std::cerr << "-s SEARCH_VAL ";
			std::cerr << "\n\n";

			std::cerr << "\tdata-file\tBinary volume data input filename\n\n";

			std::cerr << "\tDIM_TYPE\tMust be an unsigned integral type, namely unsigned (equivalent to unsigned-int), unsigned-int or unsigned-long; this datatype will be used to store the dimensions of the point grid and metacells, and thus must be large enough to accommodate their values\n\n";

			std::cerr << "\tPT_GRID_DIM_X PT_GRID_DIM_Y PT_GRID_DIM_Z\n";
			std::cerr << "\t\t\tDimensions of grid of points; note that because each point is the vertex of a cubic cell/voxel, the voxel grid thus has dimensions (PT_GRID_DIM_X - 1, PT_GRID_DIM_Y - 1, PT_GRID_DIM_Z - 1)\n";
			std::cerr << "\t\t\tDataset dimensions:\n";
			std::cerr << "\t\t\t\tIsabel: (500 500 100)\n";
			std::cerr << "\t\t\t\tRadiation: (600 248 248)\n";
			std::cerr << "\t\t\t\tTeraShake: (750 375 100)\n";
			std::cerr << "\t\t\t\tVortex: (128 128 128)\n";

			std::cerr << "\tdatatype-flag:\n";
			std::cerr << "\t\t-d, --double\tUse doubles as values\n\n";
			std::cerr << "\t\t-f, --float\tUse floats as values\n\n";
			std::cerr << "\t\t-i, --int\tUse ints for values\n\n";
			std::cerr << "\t\t-l, --long\tUse longs as values\n\n";

			std::cerr << "\ttree-type-flag:\n";
			std::cerr << "\t\t-g, --gpu\tUse StaticPSTGPU\n\n";
			std::cerr << "\t\t--iter\t\tUse StaticPSTCPUIter\n\n";
			std::cerr << "\t\t--recur\t\tUse StaticPSTCPURecur\n\n";

			std::cerr << "\t-I, --ids\tToggles assignment of IDs of data type DIM_TYPE to the nodes of the tree; defaults to false\n\n";

			std::cerr << "\t-m, --metacell-dims METACELL_DIM_X [METACELL_DIM_Y METACELL_DIM_Z]\n";
			std::cerr << "\t\t\tDimensions to use for metacells, in units of voxels^3; if only METACELL_DIM_X is provided, METACELL_DIM_Y and METACELL_DIM_Z are set to be equal to the same value; metacell dimensions default to 4 * 4 * 4 voxels^3 (as this number is both a cubic number and a multiple of 32 (warp size, for maximal thread occupancy on the GPU), and is the smallest possible metacell satisfying both those criteria, as a smaller metacell yields more metacells, a regime where PST performs well)\n\n";
			std::cerr << "\t\t\t\tCurrently only guaranteed to work for metacells of dimensions 4 * 4 * 4 voxels^3\n\n\n\n";

			std::cerr << "\t-r, --report-IDs\tWhether to report point IDs or full info of a point; defaults to full info\n\n";

			std::cerr << "\t-s, --search-val SEARCH_VAL\n";
			std::cerr << "\t\t\tValue to search for in all intervals; interval bounds are treated as inclusive\n\n";

			std::cerr << "\t-t, --timed\tToggles timing of the construction and search portion of the code; uses on-device functions for GPU PST; defaults to false\n\n";

			std::cerr << "\t-w, --warps-per-block WARPS_PER_BLOCK\n";
			std::cerr << "\t\t\tNumber of warps to use in a CUDA thread block for PST operations; only relevant when -g option is invoked; defaults to 1\n\n";

			return ExitStatusCodes::SUCCESS;
		}

		else if (i == 1)	// Check existence of data-file
		{
			if (std::filesystem::exists(arg))
				test_info.input_file = arg;
			else
			{
				std::cerr << "File " << arg << " does not exist\n";
				return ExitStatusCodes::FILE_NOT_FOUND_ERR;
			}
		}

		else if (i == 2)	// Dimension value type parsing
		{
			// Convert id_type_string to lowercase for easier processing
			std::transform(arg.begin(), arg.end(),
							arg.begin(),
							[](unsigned char c){ return std::tolower(c); });

			if (arg == "unsigned" || arg == "unsigned-int")
				test_info.id_type = DataType::UNSIGNED_INT;
			else if (arg == "unsigned-long")
				test_info.id_type = DataType::UNSIGNED_LONG;
			else
			{
				std::cerr << "Invalid argument for ID data type: " << arg << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}

		else if (i == 3)	// Get point grid dimensions
		{
			// Because of lack of parity of incrementing behavior between first and later arguments when arguments have no flags, simply use index j to access objects, and update the value of i later
			for (int j = 0; j < Dims::NUM_DIMS; j++)
			{
				if (i + j >= argc)
				{
					std::cerr << "Insufficient number of arguments provided for search bounds\n";
					return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
				}

				test_info.pt_grid_dim_strings[j] = argv[i + j];
			}
			i += Dims::NUM_DIMS - 1;
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

		// Tree type parsing
		else if (arg == "-g" || arg == "--gpu")
			test_info.tree_type = PSTType::GPU;
		else if (arg == "--iter")
			test_info.tree_type = PSTType::CPU_ITER;
		else if (arg == "--recur")
			test_info.tree_type = PSTType::CPU_RECUR;

		// ID flag
		else if (arg == "-I" || arg == "--ids")
			test_info.pts_with_ids = true;

		// Metacell dimension parsing
		else if (arg == "-m" || arg == "--metacell-dims")
		{
			for (int j = 0; j < Dims::NUM_DIMS; j++)
			{
				i++;
				if (i >= argc)
				{
					if (j == 0)		// No dimensions given
					{
						std::cerr << "Insufficient number of arguments provided for metacell dimensions\n";
						return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
					}
					else			// At least one dimension given; break out of loop
						break;
				}

				test_info.metacell_dim_strings[j] = argv[i];
			}
		}

		// Report ID flag parsing
		else if (arg == "-r" || arg == "--report-IDs")
			test_info.report_IDs = true;

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
				// Curly braces necessary around try block
				test_info.warps_per_block = std::stoul(argv[i], nullptr, 0);
			}
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for number of warps per thread block: " << argv[i] << '\n';
				return ExitStatusCodes::INVALID_ARG_ERR;
			}
		}

		// Search value parsing; as this is an interval search, this correpsonds to a two-sided left search in the dual space that transforms intervals into points in the intersections of half spaces in the real plane R^2
		else if (arg == "-s" || arg == "--search-val")
		{
			PSTTestInfoStruct::NumSearchVals num_search_vals;

			num_search_vals = PSTTestInfoStruct::NumSearchVals::NUM_VALS_TWO_SEARCH;
			test_info.test_type = PSTTestCodes::LEFT_SEARCH;

			test_info.ordered_vals = true;	// Interval search, so values in the interval must be ordered; not actually used, but set for consistency when debugging

			i++;
			if (i >= argc)
			{
				std::cerr << "Insufficient number of arguments provided for search bounds\n";
				return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
			}

			// As this is a two-sided interval search, dimension-2 value is the same as the dimension-1 value; put dimension-2 value in third slot of test_info.search_range_strings
			test_info.search_range_strings[0]
				= test_info.search_range_strings[num_search_vals] = argv[i];
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
