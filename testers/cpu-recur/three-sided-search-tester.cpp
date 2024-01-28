#include <cstdlib>
#include <iostream>
#include <random>

#include "../../point-struct.h"
#include "../../print-array.h"
#include "../../static-pst-cpu-recur.h"

int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		std::cerr << "Usage: ./search-tester num_elems min_dim1_val max_dim1_val min_dim2_val\n";
		return 1;
	}

	// If argument base == 0, strtoull() autodetects the base used for the number it is parsing
	size_t num_elems = strtoull(argv[1], nullptr, 0);

	int min_dim1_val = atoi(argv[2]);
	int max_dim1_val = atoi(argv[3]);
	int min_dim2_val = atoi(argv[4]);

	// Random number engine: Mersenne Twister 19937; takes the parameter as its seed
	std::mt19937 mt_eng(1031);
	// Distribution
	std::uniform_int_distribution<int> unif_int_dist(0,99);

	// PointStruct<int> pt_arr[4] = {PointStruct<int>(6, 17), PointStruct<int>(56, 72), PointStruct<int>(21, 63), PointStruct<int>(44, 67)};
	// PointStruct<int> pt_arr[4] = {PointStruct<int>(6, 17), PointStruct<int>(56, 72), PointStruct<int>(21, 100), PointStruct<int>(44, 67)};
	PointStruct<int> *pt_arr = new PointStruct<int>[num_elems];
	for (int i = 0; i < num_elems; i++)
	{
		pt_arr[i].dim1_val = unif_int_dist(mt_eng);
		pt_arr[i].dim2_val = unif_int_dist(mt_eng);
	}

	StaticPSTCPURecur<int> *tree = new StaticPSTCPURecur<int>(pt_arr, num_elems);
	
	if (tree != nullptr)
		std::cout << *tree << '\n';

	size_t num_res_elems = num_elems;

	PointStruct<int> *res_arr = tree->threeSidedSearch(num_res_elems, min_dim1_val, max_dim1_val, min_dim2_val);

	printArray(res_arr, 0, num_res_elems);

	delete[] pt_arr;
	delete tree;
	delete[] res_arr;

	return 0;
}
