#ifndef ISOSURFACE_DATA_PREPROCESSING_H
#define ISOSURFACE_DATA_PREPROCESSING_H

#include <fstream>
#include <type_traits>

#include "exit-status-codes.h"
#include "preprocessor-symbols.h"

template <template<typename, typename, size_t> class PointStructTemplate,
		 	size_t num_IDs, typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
PointStructTemplate<T, GridDimType, num_IDs> *formMetacells(T *vertex_arr_d, GridDimType pt_grid_dims[NUM_DIMS], GridDimType metacell_dims[NUM_DIMS])
{
	// Total number of metacells is \Pi_{i=1}^NUM_DIMS ceil((pt_grid_dims[i] - 1) / metacell_dims[i])
	// Note that the ceiling function is used because if metacell_dims[i] \not | (pt_grid_dims[i] - 1), then the last metacell(s) in dimension i will be nonempty, though not fully tiled
	// The -1 addend arises because if one surjectively assigns to each metacell the vertex on its volume that has the smallest indices in each dimension, the edges of the point grid where indices are largest in a given direction will have no metacells, as there are no further points to which to interpolate or draw such a metacell
	GridDimType num_metacells = 1;
	for (int i = 0; i < NUM_DIMS; i++)
		num_metacells *= (pt_grid_dims[i] - 1) / metacell_dims[i]
							+ ( (pt_grid_dims[i] - 1) % metacell_dims[i] == 0 ? 0 : 1);

	// TODO: CUDA allocations for PointStructTemplate<T, GridDimType, num_IDs> *

	// TODO: Call to __global__ function formMetacellsGlobal()

	return nullptr;
};

template <typename T, typename GridDimType>
	requires std::is_integral<GridDimType>::value
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
