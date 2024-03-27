#ifndef POINT_STRUCT_H
#define POINT_STRUCT_H

#include <cstring>		// To use memcpy()
#include <iostream>
#include <type_traits>	// To use static_assert()

#include "print-array.h"

// Struct to prevent instantiation of base, unspecialised version of PointStruct; templating necessary in order to defer failure of static_assert() until attempt to instantiate primary template of PointStruct
// Source:
//	https://stackoverflow.com/questions/49315890/disable-construction-of-non-fully-specialized-template-class
template <typename...>
struct deferred_false: std::false_type{};

template <typename T, typename IDType, size_t num_IDs>
struct PointStruct
{
	static_assert(deferred_false<T, IDType>::value, "Trying to instantiate PointStruct<>");
};

template <typename T>
// C++ structs differ from classes only in that structs default to public access of all members, while classes default to private access of all members
struct PointStruct<T, void, 0>
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
	inline virtual void print(std::ostream &os) const
	{
		os << '(' << dim1_val << ", " << dim2_val << ')';
	};

	// Comparison functions return < 0 if the dim1 value of this is ordered before the dim1 value of other, > 0 if the dim1 value of this is ordered after the dim1 value of other, and the result of calling comparisonTiebreaker() if dim1_val == other.dim1_val (similar statements hold true for comparison by dim2 values)
	// Declaring other as const is necessary to allow for sort comparator lambda functions to compile properly
	// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int compareDim1(const PointStruct<T, void, 0> &other) const
	{
		if (dim1_val != other.dim1_val)
			// In case of unsigned types, subtraction will never return a negative result
			return dim1_val < other.dim1_val ? -1 : 1;
		else
			return comparisonTiebreaker(other);
	};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int compareDim2(const PointStruct<T, void, 0> &other) const
	{
		if (dim2_val != other.dim2_val)
			return dim2_val < other.dim2_val ? -1 : 1;
		else
			return comparisonTiebreaker(other);
	};

	// For comparison tiebreakers, returns < 0 if memory address of this is less than memory address of other; == 0 if the memory addresses are equal (i.e. both objects are the same); > 0 if memory address of this is greater than memory address of other
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int comparisonTiebreaker(const PointStruct<T, void, 0> &other) const
	{
		return this == &other ? 0 : this < &other ? -1 : 1;
	};

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	bool operator==(const PointStruct<T, void, 0> &other) const
	{
		return dim1_val == other.dim1_val && dim2_val == other.dim2_val;
	};
};

// Specialisation of PointStruct only for these two values of num_IDs because of performance hits when accessing arrays with threads on GPU
template <typename T, typename IDType>
struct PointStruct<T, IDType, 1>
{
	// Throws a compile-time error if T is not of arithmetic (numeric) type
	// static_assert() and std::is_arithmetic are C++11 features
	// static_assert() must have two arguments to compile on CIMS
	static_assert(std::is_arithmetic<T>::value, "Input type T not of arithmetic type");
	static_assert(std::is_arithmetic<IDType>::value, "Input type IDType not of arithmetic type");

	T dim1_val;
	T dim2_val;
	IDType id;

	PointStruct()
		: dim1_val(0),
		dim2_val(0),
		id(0)
	{};

	PointStruct(IDType id)
		: dim1_val(0),
		dim2_val(0),
		id(id)
	{};

	PointStruct(T dim1_val, T dim2_val, IDType id)
		: dim1_val(dim1_val),
		dim2_val(dim2_val),
		id(id)
	{};
	// For built-in types, implicit assignment operator and implicit copy constructor default to copying bits from source to destination

	// Printing function for << printing operator to use, as private data members may be accessed in the process
	// const keyword after method name indicates that the method does not modify any data members of the associated class
	inline virtual void print(std::ostream &os) const
	{
		os << '(' << dim1_val << ", " << dim2_val << "; " << id << ')';
	};

	// Comparison functions return < 0 if the dim1 value of this is ordered before the dim1 value of other, > 0 if the dim1 value of this is ordered after the dim1 value of other, and the result of calling comparisonTiebreaker() if dim1_val == other.dim1_val (similar statements hold true for comparison by dim2 values)
	// Declaring other as const is necessary to allow for sort comparator lambda functions to compile properly
	// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int compareDim1(const PointStruct<T, IDType, 1> &other) const
	{
		if (dim1_val != other.dim1_val)
			// In case of unsigned types, subtraction will never return a negative result
			return dim1_val < other.dim1_val ? -1 : 1;
		else
			return comparisonTiebreaker(other);
	};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int compareDim2(const PointStruct<T, IDType, 1> &other) const
	{
		if (dim2_val != other.dim2_val)
			return dim2_val < other.dim2_val ? -1 : 1;
		else
			return comparisonTiebreaker(other);
	};

	// For comparison tiebreakers, returns < 0 if memory address of this is less than memory address of other; == 0 if the memory addresses are equal (i.e. both objects are the same); > 0 if memory address of this is greater than memory address of other
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int comparisonTiebreaker(const PointStruct<T, IDType, 1> &other) const
	{
		// Order by ID field value, or, if all else fails, by address in memory to guarantee stable sorting
		return id < other.id ? -1 :
					id > other.id ? 1 :
						this < &other ? -1 :
							this > &other ? 1 : 0;
	};

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	bool operator==(const PointStruct<T, IDType, 1> &other) const
	{
		return dim1_val == other.dim1_val && dim2_val == other.dim2_val && id == other.id;
	};
};

template <typename T, size_t num_IDs, typename IDType>
std::ostream &operator<<(std::ostream &os, const PointStruct<T, IDType, num_IDs> &ptstr)
{
	ptstr.print(os);
	return os;
}

#endif
