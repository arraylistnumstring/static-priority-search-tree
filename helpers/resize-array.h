#ifndef RESIZE_ARRAY_H
#define RESIZE_ARRAY_H

#include <cstring>		// To use memcpy()

// Note that one needs not use template template parameters (i.e. template parameters that take templates) if the inner, encapsulated template type is not important to the code
template <typename T>
void resizeArray(T *&pt_arr, size_t &pt_arr_size, const size_t new_pt_arr_size)
{
	T *new_pt_arr = new T[new_pt_arr_size];
	if (new_pt_arr_size < pt_arr_size)	// Shrinking array
		std::memcpy(new_pt_arr, pt_arr, new_pt_arr_size*sizeof(T));
	else	// Growing array
		std::memcpy(new_pt_arr, pt_arr, pt_arr_size*sizeof(T));

	delete[] pt_arr;

	pt_arr = new_pt_arr;
	pt_arr_size = new_pt_arr_size;
};

#endif
