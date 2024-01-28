#include <algorithm>	// To use sort()
#include <cstdlib>		// To use strtod(), strtof()
#include <cstring>		// To use strcmp()
#include <iostream>
#include <random>
#include <string>		// To use stoi()

#include "../../gpu-err-chk.h"
#include "../../err-chk.h"
#include "../../point-struct-gpu.h"
#include "../../print-array.h"
#include "../../static-pst-gpu.h"

int main(int argc, char* argv[])
{
	// Note: original testing used seed 1031
	// strcmp() returns 0 for exact matches
	if (argc < 9
			|| !(std::strcmp(argv[1], "-L") == 0
					|| std::strcmp(argv[1], "-R") == 0)
			|| !(std::strcmp(argv[2], "-d") == 0
					|| std::strcmp(argv[2], "-f") == 0
					|| std::strcmp(argv[2], "-i") == 0
					|| std::strcmp(argv[2], "-l") == 0)
	   )
	{
		std::cerr << "Usage: ./search-tester <search-type-flag> <datatype-flag> ";
		std::cerr << "min_val_range max_val_range ";
		std::cerr << "num_elems rand_seed ";
		std::cerr << "dim1_val_bound min_dim2_val\n\n";

		std::cerr << "\tsearch-type-flag:\n";
		std::cerr << "\t\t-L\tLeftwards search, i.e. treat dim1_val_bound as max dim1 value\n";
		std::cerr << "\t\t-R\tRightwards search, i.e. treat dim1_val_bound as min dim1 value\n\n";

		std::cerr << "\tdatatype-flag:\n";
		std::cerr << "\t\t-d\tUse doubles as values\n\n";
		std::cerr << "\t\t-f\tUse floats as values\n\n";
		std::cerr << "\t\t-i\tUse ints for values\n\n";
		std::cerr << "\t\t-l\tUse longs as values\n\n";
		return 1;
	}

	bool leftwards = !std::strcmp(argv[1], "-L") ? true : false;

	// If argument base == 0, strtoull() autodetects the base used for the number it is parsing
	size_t num_elems = std::strtoull(argv[5], nullptr, 0);
	size_t rand_seed = std::strtoull(argv[6], nullptr, 0);


	// When called, produces a non-deterministic 32-bit seed
	// std::random_device rand_dev;
	// Random number engine: Mersenne Twister 19937; takes the parameter as its seed
	std::mt19937 mt_eng(rand_seed);
	// std::mt19937 mt_eng(rand_dev());

	// Distribution
	if (!std::strcmp(argv[2], "-d"))
	{
		double min_val_range = std::strtod(argv[3], nullptr);
		double max_val_range = std::strtod(argv[4], nullptr);

		double dim1_val_bound = std::strtod(argv[7], nullptr);
		double min_dim2_val = std::strtod(argv[8], nullptr);

		std::uniform_real_distribution<double> unif_double_dist(min_val_range, max_val_range);

		PointStructGPU<double> *pt_arr = new PointStructGPU<double>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_double_dist(mt_eng);
			pt_arr[i].dim2_val = unif_double_dist(mt_eng);
		}
	#ifdef DEBUG
		/*
			Code that is only compiled when debugging; to define this preprocessor variable, compile with the option -DDEBUG, as in
				nvcc -DDEBUG <source-code-file>
		*/
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTGPU<double> *tree = new StaticPSTGPU<double>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		size_t num_res_elems;
		PointStructGPU<double> *res_pt_arr_d;

		if (leftwards)
			res_pt_arr_d = tree->twoSidedLeftSearch(num_res_elems, dim1_val_bound, min_dim2_val);
		else
			res_pt_arr_d = tree->twoSidedRightSearch(num_res_elems, dim1_val_bound, min_dim2_val);

		PointStructGPU<double> *res_pt_arr = new PointStructGPU<double>[num_res_elems];

		if (res_pt_arr == nullptr)
			throwErr("Error: could not allocate " + std::to_string(num_res_elems)
						+ " elements of type " + typeid(PointStructGPU<double>).name()
						+ " to root");

		gpuErrorCheck(cudaMemcpy(res_pt_arr, res_pt_arr_d, num_res_elems * sizeof(PointStructGPU<double>), cudaMemcpyDefault),
						"Error in copying result of two-sided search from device "
						+ std::to_string(tree->getDevInd()) + " of "
						+ std::to_string(tree->getNumDevs()) + ": ");

		std::sort(res_pt_arr, res_pt_arr + num_res_elems,
					[](const PointStruct<double> &pt_1, const PointStruct<double> &pt_2)
					{
						return pt_1.compareDim1(pt_2) < 0;
					});

		printArray(res_pt_arr, 0, num_res_elems);


		// As concatenatenation is left-to-right, string literals are of type const char * and typeid().name() shows up before the first instance of an std::string, it is necessary to transform either the first or second concatenation operands to type std::string
		gpuErrorCheck(cudaFree(res_pt_arr_d),
						"Error in freeing array of search results of type "
						+ std::string(typeid(PointStructGPU<double>).name()) + " on device "
						+ std::to_string(tree->getDevInd()) + " of "
						+ std::to_string(tree->getNumDevs()) + ": ");

		delete[] pt_arr;
		delete[] res_pt_arr;
		delete tree;
	}
	else if (!std::strcmp(argv[2], "-f"))
	{
		float min_val_range = std::strtof(argv[3], nullptr);
		float max_val_range = std::strtof(argv[4], nullptr);

		std::uniform_real_distribution<float> unif_float_dist(min_val_range, max_val_range);

		PointStructGPU<float> *pt_arr = new PointStructGPU<float>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_float_dist(mt_eng);
			pt_arr[i].dim2_val = unif_float_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTGPU<float> *tree = new StaticPSTGPU<float>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}
	else if (!std::strcmp(argv[2], "-i"))
	{
		int min_val_range = std::stoi(argv[3], nullptr, 0);
		int max_val_range = std::stoi(argv[4], nullptr, 0);

		std::uniform_int_distribution<int> unif_int_dist(min_val_range, max_val_range);

		PointStructGPU<int> *pt_arr = new PointStructGPU<int>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_int_dist(mt_eng);
			pt_arr[i].dim2_val = unif_int_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTGPU<int> *tree = new StaticPSTGPU<int>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}
	else	// std::strcmp(argv[2], "-l") == 0
	{
		long min_val_range = std::stol(argv[3], nullptr, 0);
		long max_val_range = std::stol(argv[4], nullptr, 0);

		std::uniform_int_distribution<long> unif_long_dist(min_val_range, max_val_range);

		PointStructGPU<long> *pt_arr = new PointStructGPU<long>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_long_dist(mt_eng);
			pt_arr[i].dim2_val = unif_long_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTGPU<long> *tree = new StaticPSTGPU<long>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}

	return 0;
}
