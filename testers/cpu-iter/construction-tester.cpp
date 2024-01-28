#include <cstdlib>	// To use strtod(), strtof()
#include <cstring>	// To use strcmp()
#include <iostream>
#include <random>
#include <string>	// To use stoi()

#include "../../point-struct-cpu-iter.h"
#include "../../static-pst-cpu-iter.h"

#ifdef DEBUG
/*
	Code that is only compiled when debugging; to define this preprocessor variable, compile with the option -DDEBUG, as in
		nvcc -DDEBUG <source-code-file>
*/
#include "../../print-array.h"
#endif

int main(int argc, char* argv[])
{
	// Note: original testing used seed 1031
	// strcmp() returns 0 for exact matches
	if (argc < 6
			|| !(std::strcmp(argv[1], "-d") == 0
					|| std::strcmp(argv[1], "-f") == 0
					|| std::strcmp(argv[1], "-i") == 0
					|| std::strcmp(argv[1], "-l") == 0)
	   )
	{
		std::cerr << "Usage: ./construction-tester <datatype-flag> min_val_range max_val_range num_elems rand_seed\n";
		std::cerr << "\tOptions:\n";
		std::cerr << "\t\t-d\tUse doubles as values\n\n";
		std::cerr << "\t\t-f\tUse floats as values\n\n";
		std::cerr << "\t\t-i\tUse ints for values\n\n";
		std::cerr << "\t\t-l\tUse longs as values\n\n";
		return 1;
	}

	// If argument base == 0, strtoull() autodetects the base used for the number it is parsing
	size_t num_elems = std::strtoull(argv[4], nullptr, 0);
	size_t rand_seed = std::strtoull(argv[5], nullptr, 0);

	// When called, produces a non-deterministic 32-bit seed
	// std::random_device rand_dev;
	// Random number engine: Mersenne Twister 19937; takes the parameter as its seed
	std::mt19937 mt_eng(rand_seed);
	// std::mt19937 mt_eng(rand_dev());

	// Distribution
	if (!std::strcmp(argv[1], "-d"))
	{
		double min_val_range = std::strtod(argv[2], nullptr);
		double max_val_range = std::strtod(argv[3], nullptr);

		std::uniform_real_distribution<double> unif_double_dist(min_val_range, max_val_range);

		PointStructCPUIter<double> *pt_arr = new PointStructCPUIter<double>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_double_dist(mt_eng);
			pt_arr[i].dim2_val = unif_double_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTCPUIter<double> *tree = new StaticPSTCPUIter<double>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}
	else if (!std::strcmp(argv[1], "-f"))
	{
		float min_val_range = std::strtof(argv[2], nullptr);
		float max_val_range = std::strtof(argv[3], nullptr);

		std::uniform_real_distribution<float> unif_float_dist(min_val_range, max_val_range);

		PointStructCPUIter<float> *pt_arr = new PointStructCPUIter<float>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_float_dist(mt_eng);
			pt_arr[i].dim2_val = unif_float_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTCPUIter<float> *tree = new StaticPSTCPUIter<float>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}
	else if (!std::strcmp(argv[1], "-i"))
	{
		int min_val_range = std::stoi(argv[2], nullptr, 0);
		int max_val_range = std::stoi(argv[3], nullptr, 0);

		std::uniform_int_distribution<int> unif_int_dist(min_val_range, max_val_range);

		PointStructCPUIter<int> *pt_arr = new PointStructCPUIter<int>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_int_dist(mt_eng);
			pt_arr[i].dim2_val = unif_int_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTCPUIter<int> *tree = new StaticPSTCPUIter<int>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}
	else	// std::strcmp(argv[1], "-l") == 0
	{
		long min_val_range = std::stol(argv[2], nullptr, 0);
		long max_val_range = std::stol(argv[3], nullptr, 0);

		std::uniform_int_distribution<long> unif_long_dist(min_val_range, max_val_range);

		PointStructCPUIter<long> *pt_arr = new PointStructCPUIter<long>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = unif_long_dist(mt_eng);
			pt_arr[i].dim2_val = unif_long_dist(mt_eng);
		}
	#ifdef DEBUG
		printArray(pt_arr, 0, num_elems);
	#endif

		StaticPSTCPUIter<long> *tree = new StaticPSTCPUIter<long>(pt_arr, num_elems);
		
		if (tree != nullptr)
			std::cout << *tree << '\n';

		delete[] pt_arr;
		delete tree;
	}

	return 0;
}
