#ifndef PRINT_ARRAY_H
#define PRINT_ARRAY_H

template <typename T>
void printArray(T *const &T_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
void printArrayDereferenced(T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
void printArrayIndexed(T *const &T_arr, size_t *const &ind_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
void printArrayOffsetFromStart(T *const &T_arr, T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems);

template <typename T>
void printArray(T *const &T_arr, const size_t start_ind, const size_t num_elems)
{
	std::cout << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		std::cout << T_arr[i] << ' ';
	std::cout << "]\n";
}

// For printing the content referenced by an array of pointers
template <typename T>
void printArrayDereferenced(T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems)
{
	std::cout << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		std::cout << *T_ptr_arr[i] << ' ';
	std::cout << "]\n";
}


// For printing one array using another's elements as indices
template <typename T>
void printArrayIndexed(T *const &T_arr, size_t *const &ind_arr, const size_t start_ind, size_t num_elems)
{
	std::cout << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		std::cout << T_arr[ind_arr[i]] << ' ';
	std::cout << "]\n";
}

// For printing the indices of one array according to the ordering provided by an array of pointers to it
template <typename T>
void printArrayOffsetFromStart(T *const &T_arr, T **const &T_ptr_arr, const size_t start_ind, const size_t num_elems)
{
	std::cout << "[ ";
	for (size_t i = start_ind; i < start_ind + num_elems; i++)
		std::cout << (T_ptr_arr[i] - T_arr) << ' ';
	std::cout << "]\n";
}

#endif
