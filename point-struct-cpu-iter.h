#ifndef POINT_STRUCT_CPU_ITER_H
#define POINT_STRUCT_CPU_ITER_H

#include "point-struct.h"

template <typename T>
// C++ structs differ from classes only in that structs default to public access of all members, while classes default to private access of all members
// Inheritance in order for comparison functions to be executable as inline functions on GPU
struct PointStructCPUIter : public PointStruct<T>
{
	// Comparison functions return < 0 if the dim1 value of this is ordered before the dim1 value of other, > 0 if the dim1 value of this is ordered after the dim1 value of other, and the result of calling comparisonTiebreaker() if dim1_val == other.dim1_val (similar statements hold true for comparison by dim2 values)
	// Static wrapper comparison function; allows use of comparison functions without said function being tied to a particular object in the way that member functions are
	// Static functions cannot be const, as they have no object bound to keyword this that they would refrain from modifying
	static int compareDim1(PointStructCPUIter<T> &pt1, PointStructCPUIter<T> &pt2)
		{return pt1.compareDim1(pt2);};
	int compareDim1(PointStructCPUIter<T> &other) const
	{
		// this->dim1_val is necessary in a template-derived subclass, as dim1_val on its own is a non-dependent name (i.e. not dependent on the template T), while PointStruct<T> is a dependent name (because it is dependent on the template T). Hence, compilers do not look in dependent base classes (e.g. PointStruct<T>) when looking up non-dependent names (e.g. dim1_val); this->dim1_val turns dim1_val into a dependent name, and therefore resolves names in the desired fashion; the dereference operator then goes to the memory address specified by this->dim1_val
		// Source:
		// https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
		if (this->dim1_val != other.dim1_val)
			// In case of unsigned types, subtraction will never return a negative result
			return this->dim1_val < other.dim1_val ? -1 : 1;
		else
			return this->comparisonTiebreaker(other);
	};
	static int compareDim2(PointStructCPUIter<T> &pt1, PointStructCPUIter<T> &pt2)
		{return pt1.compareDim2(pt2);};
	int compareDim2(PointStructCPUIter<T> &other) const
	{
		if (this->dim2_val != other.dim2_val)
			return this->dim2_val < other.dim2_val ? -1 : 1;
		else
			return this->comparisonTiebreaker(other);
	};

	// For comparison tiebreakers, returns < 0 if memory address of this is less than memory address of other; == 0 if the memory addresses are equal (i.e. both objects are the same); > 0 if memory address of this is greater than memory address of other
	int comparisonTiebreaker(PointStructCPUIter<T> &other) const
	{
		return this == &other ? 0 : this < &other ? -1 : 1;
	};

	bool operator==(PointStructCPUIter<T> &other) const
	{
		return this->dim1_val == other.dim1_val && this->dim2_val == other.dim2_val;
	};
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const PointStructCPUIter<T> &ptstr)
{
	ptstr.print(os);
	return os;
}

#endif
