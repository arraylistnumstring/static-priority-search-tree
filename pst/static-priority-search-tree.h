#ifndef STATIC_PRIORITY_SEARCH_TREE_H
#define STATIC_PRIORITY_SEARCH_TREE_H

#include <ostream>


template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType=void, size_t num_IDs=0>
class StaticPrioritySearchTree
{
	public:
		// = 0 indicates that this is a pure virtual function, i.e. defines an interface strictly for subclasses to implement
		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const = 0;

		// Search functions, though necessary, are not made to be virtual functions here, as virtual functions with template return types are not allowed (as the vtable, a lookup table with which virtual functions are often implemented, would necessarily be potentially infinite, to allow for all the potential variations and inheritances of any possible template type)
};

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			template<typename, template<typename, typename, size_t> class, typename, size_t> class StaticPrioritySearchTree,
			typename IDType, size_t num_IDs
		 >
std::ostream &operator<<(std::ostream &os, StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_IDs> &pst)
{
	pst.print(os);
	return os;
}

#endif
