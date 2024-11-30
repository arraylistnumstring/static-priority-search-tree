#ifndef BINARY_INPUT_FILE_READER_H
#define BINARY_INPUT_FILE_READER_H

#include <fstream>

#include "preprocessor-symbols.h"

template <typename T, typename GridDimType>
T *readInVertices(std::string input_filename, GridDimType grid_dims[NUM_DIMS])
{
	GridDimType num_vertices = 1;
	for (int i = 0; i < NUM_DIMS; i++)
		num_vertices *= grid_dims[i];

	T *vertex_arr = new T[num_vertices];

	std::ifstream input_filestream {input_filename, std::ios_base::binary};

	input_filestream.read()
};

#endif
