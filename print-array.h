#ifndef PRINT_ARRAY_H
#define PRINT_ARRAY_H

#include <iostream>

template <typename T>
std::ostream &printArray(std::ostream &os, T *const &T_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
std::ostream &printArrayDereferenced(std::ostream &os, T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
std::ostream &printArrayIndexed(std::ostream &os, T *const &T_arr, size_t *const &ind_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
std::ostream &printArrayOffsetFromStart(std::ostream &os, T *const &T_arr, T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
std::ostream &printArray(std::ostream &os, T *const &T_arr, const size_t start_ind, const size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << T_arr[i] << ' ';
	os << ']';
	return os;
}

// For printing the content referenced by an array of pointers
template <typename T>
std::ostream &printArrayDereferenced(std::ostream &os, T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << *T_ptr_arr[i] << ' ';
	os << ']';
	return os;
}


// For printing one array using another's elements as indices
template <typename T>
std::ostream &printArrayIndexed(std::ostream &os, T *const &T_arr, size_t *const &ind_arr, const size_t start_ind, size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << T_arr[ind_arr[i]] << ' ';
	os << ']';
	return os;
}

// For printing the indices of one array according to the ordering provided by an array of pointers to it
template <typename T>
std::ostream &printArrayOffsetFromStart(std::ostream &os, T *const &T_arr, T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems)
{
	os << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		os << (T_ptr_arr[i] - T_arr) << ' ';
	os << ']';
	return os;
}

// Allows printArray*() to be the second operand to general first-operand ostreams
template <typename... Types>	// typename... Types is a parameter pack of types
// Types... expands the parameter pack defined above into its constituent types
inline std::ostream &operator<<(std::basic_ostream<Types...> &os, std::ostream &os2)
{
	return os << os2;
}

#endif
