#ifndef BINARY_INPUT_FILE_READER_H
#define BINARY_INPUT_FILE_READER_H

#include <fstream>

#include "exit-status-codes.h"
#include "preprocessor-symbols.h"

template <typename T, typename GridDimType>
T *readInVertices(std::string input_filename, GridDimType grid_dims[NUM_DIMS])
{
	GridDimType num_vertices = 1;
	for (int i = 0; i < NUM_DIMS; i++)
		num_vertices *= grid_dims[i];

	T *vertex_arr = new T[num_vertices];

	std::ifstream input_filestream(input_filename, std::ios_base::binary);

	if (!input_filestream.is_open())	// Check that filestream is open
	{
		std::cerr << "Failed to open " << input_filename << " for reading\n";
		std::exit(ExitStatusCodes::FILE_OPEN_ERR);
	}

	input_filestream.read(reinterpret_cast<char *>(vertex_arr), num_vertices * sizeof(T));

	// Check that the appropriate number of bytes were read
	if (input_filestream.gcount() != num_vertices * sizeof(T))
	{
		std::cerr << "Failure: read " << input_filestream.gcount() << " bytes, expected " << num_vertices * sizeof(T) << " bytes\n";
		std::exit(ExitStatusCodes::INPUT_READ_ERR);
	}

	return vertex_arr;
};

#endif
