#ifndef STATIC_PRIORITY_SEARCH_TREE_H
#define STATIC_PRIORITY_SEARCH_TREE_H

#include <concepts>
#include <iostream>
#include <type_traits>	// To filter out non-numeric types of T

#include "point-struct.h"

// Use concepts, a C++20 feature, to determine that the provided PtStruct type is valid
template <typename T, typename PtStruct<typename>>
concept ValidPtStruct = requires(T t, PtStruct<T> ptstr)
{
	ptstr.dim1_val;
	ptstr.dim2_val;
	ptstr.compareDim1(ptstr);
	ptstr.compareDim2(ptstr);
};

template <typename T, size_t num_ID_fields=0, typename IDType=void, typename PtStruct<typename, size_t, typename>>
class StaticPrioritySearchTree	// abstract class
{
	// Throws a compile-time error if T is not of arithmetic (numeric) type
	// static_assert() and std::is_arithmetic are C++11 features
	// static_assert() must have two arguments to compile on CIMS
	static_assert(std::is_arithmetic<T>::value, "Input type T not of arithmetic type");
	static_assert((num_ID_fields == 0) || (num_ID_fields > 0 && !std::is_void(IDType), "num_ID_fields must have value 0 or be of non-zero value with IDType being of non-void type")

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
