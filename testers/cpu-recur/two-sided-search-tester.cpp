#include <algorithm>	// To use sort()
#include <cstdlib>		// To use strtod(), strtof()
#include <cstring>		// To use strcmp()
#include <iostream>
#include <random>
#include <string>		// To use stoi()

#include "../../point-struct.h"
#include "../../print-array.h"
#include "../../static-pst-cpu-recur.h"

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

		PointStruct<double> *pt_arr = new PointStruct<double>[num_elems];

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
		printArray(std::cout, pt_arr, 0, num_elems);
		std::cout << '\n';
	#endif

		StaticPSTCPURecur<double> *tree = new StaticPSTCPURecur<double>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		size_t num_res_elems = 0;
		PointStruct<double> *res_pt_arr;

		if (leftwards)
			res_pt_arr = tree->twoSidedLeftSearch(num_res_elems, dim1_val_bound, min_dim2_val);
		else
			res_pt_arr = tree->twoSidedRightSearch(num_res_elems, dim1_val_bound, min_dim2_val);

		std::sort(res_pt_arr, res_pt_arr + num_res_elems,
					[](const PointStruct<double> &pt_1, const PointStruct<double> &pt_2)
					{
						return pt_1.compareDim1(pt_2) < 0;
					});

		printArray(std::cout, res_pt_arr, 0, num_res_elems);
		std::cout << '\n';

		delete[] pt_arr;
		delete[] res_pt_arr;
		delete tree;
	}
	else if (!std::strcmp(argv[2], "-f"))
	{
		float min_val_range = std::strtof(argv[3], nullptr);
		float max_val_range = std::strtof(argv[4], nullptr);

		std::uniform_real_distribution<float> unif_float_dist(min_val_range, max_val_range);

		PointStruct<float> *pt_arr = new PointStruct<float>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_float_dist(mt_eng);
			pt_arr[i].dim2_val = unif_float_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(std::cout, pt_arr, 0, num_elems);
		std::cout << '\n';
	#endif

		StaticPSTCPURecur<float> *tree = new StaticPSTCPURecur<float>(pt_arr, num_elems);
		
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

		PointStruct<int> *pt_arr = new PointStruct<int>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_int_dist(mt_eng);
			pt_arr[i].dim2_val = unif_int_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(std::cout, pt_arr, 0, num_elems);
		std::cout << '\n';
	#endif

		StaticPSTCPURecur<int> *tree = new StaticPSTCPURecur<int>(pt_arr, num_elems);
		
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

		PointStruct<long> *pt_arr = new PointStruct<long>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_long_dist(mt_eng);
			pt_arr[i].dim2_val = unif_long_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(std::cout, pt_arr, 0, num_elems);
		std::cout << '\n';
	#endif

		StaticPSTCPURecur<long> *tree = new StaticPSTCPURecur<long>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}

	return 0;
}
