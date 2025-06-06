#ifndef PRINT_ARRAY_H
#define PRINT_ARRAY_H

#include <ostream>

#include "linearise-id.h"


template <typename T, typename GridDimType>
std::ostream &print3DArray(std::ostream &os, T *const T_arr,
							const GridDimType start_inds[Dims::NUM_DIMS],
							const GridDimType grid_dims[Dims::NUM_DIMS])
{
	for (GridDimType k = start_inds[Dims::Z_DIM_IND]; k < grid_dims[Dims::Z_DIM_IND]; k++)
		for (GridDimType j = start_inds[Dims::Y_DIM_IND]; j < grid_dims[Dims::Y_DIM_IND]; j++)
			for (GridDimType i = start_inds[Dims::X_DIM_IND]; i < grid_dims[Dims::X_DIM_IND]; i++)
			{
				os << 'V';
				os << '[' << i << ']';
				os << '[' << j << ']';
				os << '[' << k << ']';
				os << '=' << T_arr[lineariseID(i, j, k, grid_dims[Dims::X_DIM_IND], grid_dims[Dims::Y_DIM_IND])];
				os << '\n';
			}

	return os;
};

template <typename T>
std::ostream &printArray(std::ostream &os, T *const T_arr, const size_t start_ind, const size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << T_arr[i] << ' ';
	os << ']';
	return os;
};

// For printing the content referenced by an array of pointers
template <typename T>
std::ostream &printArrayDereferenced(std::ostream &os, T **const T_ptr_arr, const size_t start_ind, const size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << *T_ptr_arr[i] << ' ';
	os << ']';
	return os;
};


// For printing one array using another's elements as indices
template <typename T>
std::ostream &printArrayIndexed(std::ostream &os, T *const T_arr, size_t *const ind_arr, const size_t start_ind, size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << T_arr[ind_arr[i]] << ' ';
	os << ']';
	return os;
};

// For printing the indices of one array according to the ordering provided by an array of pointers to it
template <typename T>
std::ostream &printArrayOffsetFromStart(std::ostream &os, T *const T_arr, T **const T_ptr_arr, const size_t start_ind, const size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << (T_ptr_arr[i] - T_arr) << ' ';
	os << ']';
	return os;
};

#endif
