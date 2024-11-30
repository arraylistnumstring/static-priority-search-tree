#include <filesystem>	// To use filesystem existence checks
#include <random>		// To use uniform_int_distribution and uniform_real_distribution
#include <string>		// To use stoi() and string operators for command-line argument parsing

#include "exit-status-codes.h"		// For consistent exit status codes
#include "ips-test-info-struct.h"

int main(int argc, char *argv[])
{
#ifdef DEBUG_TEST
	std::cout << "Began interval parallel search tester driver\n";
#endif
	IPSTestInfoStruct test_info;

	// Parse command-line arguments
	for (int i = 0; i < argc; i++)
	{
		std::string arg(argv[i]);	// Allow use of string's operators and functions

		// Help message
		if (arg == "-h" || arg == "--help")
		{
			std::cerr << "Usage: ./ips-dataset-tester-driver data-file ";
			std::cerr << "ID_TYPE PT_GRID_DIM_X PT_GRID_DIM_Y PT_GRID_DIM_Z ";
			std::cerr << "<datatype-flag> ";
			std::cerr << "[-r] ";
			std::cerr << "[-t] ";
			std::cerr << "-B NUM_BLOCKS";
			std::cerr << "-m METACELL_DIM_X [METACELL_DIM_Y METACELL_DIM_Z] ";
			std::cerr << "-s SEARCH_VAL ";
			std::cerr << "-T THREADS_PER_BLOCK";
			std::cerr << "\n\n";

			std::cerr << "\tdata-file:\tBinary volume data input filename\n\n";

			std::cerr << "\tID_TYPE\tMust be an unsigned integral type, namely unsigned (equivalent to unsigned-int), unsgiend-int or unsigned-long; this datatype will be used to store the dimensions of the point grid and metacells, and thus must be large enough to accommodate their values\n\n";

			std::cerr << "\tPT_GRID_DIM_X PT_GRID_DIM_Y PT_GRID_DIM_Z:\n";
			std::cerr << "\t\tDimensions of grid of points; note that because each point is the vertex of a cubic cell/voxel, the voxel grid thus has dimensions (PT_GRID_DIM_X - 1, PT_GRID_DIM_Y - 1, PT_GRID_DIM_Z - 1)\n\n";

			std::cerr << "\tdatatype-flag:\n";
			std::cerr << "\t\t-d, --double\tUse doubles as values\n\n";
			std::cerr << "\t\t-f, --float\tUse floats as values\n\n";
			std::cerr << "\t\t-i, --int\tUse ints for values\n\n";
			std::cerr << "\t\t-l, --long\tUse longs as values\n\n";

			std::cerr << "\t-B, --num-blocks NUM_BLOCKS\tNumber of blocks to use in grid for CUDA kernel\n\n";

			std::cerr << "\t-m, --metacell-dims METACELL_DIM_X [METACELL_DIM_Y METACELL_DIM_Z]\n";
			std::cerr << "\t\tDimensions to use for metacells, in units of voxels^3; if only METACELL_DIM_X is provided, METACELL_DIM_Y and METACELL_DIM_Z are set to be equal to the same value; metacell dimensions default to 4 * 4 * 4 voxels^3 (as this number is both a cubic number and a multiple of 32 (warp size, for maximal thread occupancy on the GPU), and is the smallest possible metacell satisfying both those criteria, as a smaller metacell yields more metacells, a regime where PST performs well)";

			std::cerr << "\t-r, --report-IDs\tWhether to report point IDs or full info of a point; defaults to full info\n\n";

			std::cerr << "\t-s, --search-val SEARCH_VAL\tValue to search for in all intervals; interval bounds are treated as inclusive\n\n";

			std::cerr << "\t-T, --threads-per-block THREADS_PER_BLOCK\tNumber of threads to use in a thread block\n\n";

			std::cerr << "\t-t, --timed-CUDA\tToggles timing of the CUDA portion of the code using on-device functions; defaults to false\n\n";

			return ExitStatusCodes::SUCCESS;
		}

		else if (i == 1)	// Check existence of data file
		{
			if (std::filesystem::exists(arg))
				test_info.input_file = arg;
			else
			{
				std::cerr << "File " << arg << " does not exist\n";
				return ExitStatusCodes::FILE_NOT_FOUND_ERR;
			}
		}

		else if (i == 2)	// ID type parsing
		{
			test_info.pts_with_ids = true;

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
			for (int j = 0; j < IPSTestInfoStruct::NUM_DIMS; j++)
			{
				if (i >= argc)
				{
					std::cerr << "Insufficient number of arguments provided for search bounds\n";
					return ExitStatusCodes::INSUFFICIENT_NUM_ARGS_ERR;
				}

				test_info.pt_grid_dim_strings[i] = argv[i];
				i++;
			}
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

		// Metacell dimension parsing
		else if (arg == "-m" || arg == "--metacell-dims")
		{
			for (int j = 0; j < IPSTestInfoStruct::NUM_DIMS; j++)
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
				// Curly braces necessary around try blocks
				test_info.num_thread_blocks = std::stoul(argv[i]);
			}
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for number of thread blocks: " << argv[i] << '\n';
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
			catch (std::invalid_argument const &ex)
			{
				std::cerr << "Invalid argument for number of threads per block: " << argv[i] << '\n';
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
