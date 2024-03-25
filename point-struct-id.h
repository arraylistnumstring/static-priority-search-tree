#ifndef POINT_STRUCT_ID_H
#define POINT_STRUCT_ID_H

#include "point-struct.h"
#include "print-array.h"

template <typename T, size_t num_ID_fields, typename IDType>
// C++ structs differ from classes only in that structs default to public access of all members, while classes default to private access of all members
struct PointStructID : public PointStruct<T>
{
	// Throws a compile-time error if the condition is not met
	// static_assert() is a C++11 feature
	// static_assert() must have two arguments to compile on CIMS
	static_assert(num_ID_fields > 0, "Number of ID fields num_ID_fields not greater than 0");

	IDType ids[num_ID_fields];

	// Use of (&ids)[num_ID_fields] forces passed-in array reference to have length num_ID_fields
	PointStructID(IDType (&ids)[num_ID_fields])
		: PointStruct<T>()		// Constructor delegation prevents other parameters from being initialised with member initialiser lists
	{
		for (size_t i = 0; i < num_ID_fields; i++)
			this->ids[i] = ids[i];
	};

	PointStructID(T dim1_val, T dim2_val, IDType (&ids)[num_ID_fields])
		: PointStruct<T>(dim1_val, dim2_val)
	{
		for (size_t i = 0; i < num_ID_fields; i++)
			this->ids[i] = ids[i];
	};
	// For built-in types, implicit assignment operator and implicit copy constructor default to copying bits from source to destination

	// Printing function for << printing operator to use, as private data members may be accessed in the process
	// const keyword after method name indicates that the method does not modify any data members of the associated class
	inline virtual void print(std::ostream &os) const
	{
		// this->dim1_val is necessary in a template-derived subclass, as dim1_val on its own is a non-dependent name (i.e. not dependent on the template T), while PointStruct<T> is a dependent name (because it is dependent on the template T). Hence, compilers do not look in dependent base classes (e.g. PointStruct<T>) when looking up non-dependent names (e.g. dim1_val); this->dim1_val turns dim1_val into a dependent name, and therefore resolves names in the desired fashion; the dereference operator then goes to the memory address specified by this->dim1_val
		// Source:
		// https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
		os << '(' << this->dim1_val << ", " << this->dim2_val << "; " << printArray(os, ids, 0, num_ID_fields) << ')';
	};

	// Comparison functions return < 0 if the dim1 value of this is ordered before the dim1 value of other, > 0 if the dim1 value of this is ordered after the dim1 value of other, and the result of calling comparisonTiebreaker() if this->dim1_val == other.dim1_val (similar statements hold true for comparison by dim2 values)
	// Declaring other as const is necessary to allow for sort comparator lambda functions to compile properly
	// Preprocessor directives to add keywords based on whether CUDA GPU support is available; __CUDA_ARCH__ is either undefined or defined as 0 in host code
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int compareDim1(const PointStructID<T, num_ID_fields, IDType> &other) const
	{
		if (this->dim1_val != other.dim1_val)
			// In case of unsigned types, subtraction will never return a negative result
			return this->dim1_val < other.dim1_val ? -1 : 1;
		else
			return comparisonTiebreaker(other);
	};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int compareDim2(const PointStructID<T, num_ID_fields, IDType> &other) const
	{
		if (this->dim2_val != other.dim2_val)
			return this->dim2_val < other.dim2_val ? -1 : 1;
		else
			return comparisonTiebreaker(other);
	};

	// For comparison tiebreakers, returns < 0 if memory address of this is less than memory address of other; == 0 if the memory addresses are equal (i.e. both objects are the same); > 0 if memory address of this is greater than memory address of other
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	int comparisonTiebreaker(const PointStructID<T, num_ID_fields, IDType> &other) const
	{
		// Order by successive ID field values
		for (size_t i = 0; i < num_ID_fields; i++)
		{
			if (ids[i] < other.ids[i])
				return -1;
			else if (ids[i] > other.ids[i])
				return 1;
		}

		// If all else fails, order by address in memory to guarantee stable sorting
		return this == &other ? 0 : this < &other ? -1 : 1;
	};

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
	__forceinline__ __host__ __device__
#else
	inline
#endif
	bool operator==(const PointStructID<T, num_ID_fields, IDType> &other) const
	{
		// Check that ID values all agree
		for (size_t i = 0; i < num_ID_fields; i++)
			if (ids[i] != other.ids[i])
				return false;

		return this->dim1_val == other.dim1_val && this->dim2_val == other.dim2_val;
	};
};

template <typename T, size_t num_ID_fields, typename IDType>
std::ostream &operator<<(std::ostream &os, const PointStructID<T, num_ID_fields, IDType> &ptstr)
{
	ptstr.print(os);
	return os;
}

#endif
