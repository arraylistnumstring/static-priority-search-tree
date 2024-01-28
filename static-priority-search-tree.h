#ifndef STATIC_PRIORITY_SEARCH_TREE_H
#define STATIC_PRIORITY_SEARCH_TREE_H

#include <iostream>
#include <type_traits>	// To filter out non-numeric types of T

#include "point-struct.h"

template <typename T>
class StaticPrioritySearchTree	// abstract class
{
	// Throws a compile-time error if T is not of arithmetic (numeric) type
	// static_assert() and std::is_arithmetic are C++11 features
	// static_assert() must have two arguments to compile on CIMS
	static_assert(std::is_arithmetic<T>::value, "Input type T not of arithmetic type");

	public:
		// = 0 indicates that this is a pure virtual function, i.e. defines an interface strictly for subclasses to implement
		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const = 0;
		virtual PointStruct<T>* threeSidedSearch(size_t &num_res_elems, T min_dim1_val, T max_dim1_val, T min_dim2_val) = 0;
		virtual PointStruct<T>* twoSidedLeftSearch(size_t &num_res_elems, T max_dim1_val, T min_dim2_val) = 0;
		virtual PointStruct<T>* twoSidedRightSearch(size_t &num_res_elems, T min_dim1_val, T min_dim2_val) = 0;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, StaticPrioritySearchTree<T> &t)
{
	t.print(os);
	return os;
}

#endif
