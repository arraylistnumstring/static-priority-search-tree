#ifndef STATIC_PRIORITY_SEARCH_TREE_H
#define STATIC_PRIORITY_SEARCH_TREE_H

#include <iostream>

#include "point-struct.h"

template <class PointStructClass>
class StaticPrioritySearchTree	// abstract class
{
	public:
		// = 0 indicates that this is a pure virtual function, i.e. defines an interface strictly for subclasses to implement
		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const = 0;
		virtual PointStructClass* threeSidedSearch(size_t &num_res_elems, T min_dim1_val, T max_dim1_val, T min_dim2_val) = 0;
		virtual PointStructClass* twoSidedLeftSearch(size_t &num_res_elems, T max_dim1_val, T min_dim2_val) = 0;
		virtual PointStructClass* twoSidedRightSearch(size_t &num_res_elems, T min_dim1_val, T min_dim2_val) = 0;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, StaticPrioritySearchTree<T> &t)
{
	t.print(os);
	return os;
}

#endif
