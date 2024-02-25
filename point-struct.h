#ifndef POINT_STRUCT_H
#define POINT_STRUCT_H

#include <cstring>		// To use memcpy()

template <typename T>
// C++ structs differ from classes only in that structs default to public access of all members, while classes default to private access of all members
struct PointStruct
{
	// Throws a compile-time error if T is not of arithmetic (numeric) type
	// static_assert() and std::is_arithmetic are C++11 features
	// static_assert() must have two arguments to compile on CIMS
	static_assert(std::is_arithmetic<T>::value, "Input type T not of arithmetic type");

	T dim1_val;
	T dim2_val;

	PointStruct()
		: dim1_val(0),
		dim2_val(0)
	{};

	PointStruct(T dim1_val, T dim2_val)
		: dim1_val(dim1_val),
		dim2_val(dim2_val)
	{};
	// For built-in types, implicit assignment operator and implicit copy constructor default to copying bits from source to destination

	// Printing function for << printing operator to use, as private data members may be accessed in the process
	// const keyword after method name indicates that the method does not modify any data members of the associated class
	inline void print(std::ostream &os) const
	{
		os << '(' << dim1_val << ", " << dim2_val << ')';
	};

	// Comparison functions return < 0 if the dim1 value of this is ordered before the dim1 value of other, > 0 if the dim1 value of this is ordered after the dim1 value of other, and the result of calling comparisonTiebreaker() if dim1_val == other.dim1_val (similar statements hold true for comparison by dim2 values)
	// Declaring other as const is necessary to allow for sort comparator lambda functions to compile properly
	inline int compareDim1(const PointStruct<T> &other) const
	{
		if (dim1_val != other.dim1_val)
			// In case of unsigned types, subtraction will never return a negative result
			return this->dim1_val < other.dim1_val ? -1 : 1;
		else
			return this->comparisonTiebreaker(other);
	};
	inline int compareDim2(const PointStruct<T> &other) const
	{
		if (dim2_val != other.dim2_val)
			return this->dim2_val < other.dim2_val ? -1 : 1;
		else
			return this->comparisonTiebreaker(other);
	};

	// For comparison tiebreakers, returns < 0 if memory address of this is less than memory address of other; == 0 if the memory addresses are equal (i.e. both objects are the same); > 0 if memory address of this is greater than memory address of other
	inline int comparisonTiebreaker(const PointStruct<T> &other) const
	{
		return this == &other ? 0 : this < &other ? -1 : 1;
	};

	inline bool operator==(const PointStruct<T> &other) const
	{
		return dim1_val == other.dim1_val && dim2_val == other.dim2_val;
	};
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const PointStruct<T> &ptstr)
{
	ptstr.print(os);
	return os;
}

#endif
