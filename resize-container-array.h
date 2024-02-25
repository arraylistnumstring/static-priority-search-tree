#ifndef RESIZE_CONTAINER_ARRAY_H
#define RESIZE_CONTAINER_ARRAY_H

#include <cstring>		// To use memcpy()

// Before C++17, template template parameters (i.e. template parameters that take templates) must be declared with the keyword template<...> class, rather than template typename
template <typename T, template <typename> class Container>
void resizeContainerArray(Container<T> *&pt_arr, size_t &pt_arr_size, const size_t new_pt_arr_size)
{
	Container<T> *new_pt_arr = new Container<T>[new_pt_arr_size];
	if (new_pt_arr_size < pt_arr_size)	// Shrinking array
		std::memcpy(new_pt_arr, pt_arr, new_pt_arr_size*sizeof(Container<T>));
	else	// Growing array
		std::memcpy(new_pt_arr, pt_arr, pt_arr_size*sizeof(Container<T>));

	delete[] pt_arr;

	pt_arr = new_pt_arr;
	pt_arr_size = new_pt_arr_size;
}

#endif
