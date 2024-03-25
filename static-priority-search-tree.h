#ifndef STATIC_PRIORITY_SEARCH_TREE_H
#define STATIC_PRIORITY_SEARCH_TREE_H

#include <concepts>		// To create abstract class template; requires C++20
#include <iostream>

template <typename T, template<typename, size_t, typename> class PointStructTemplate,
			template<typename, template<typename, size_t, typename> class, size_t, typename> class StaticPrioritySearchTree,
			size_t num_ID_fields=0, typename IDType=void
		 >
concept StaticPrioritySearchTree = requires (T t, PointStructTemplate<T, num_ID_fields, IDType> ptstr,
												StaticPrioritySearchTree<T, PointStructTemplate, num_ID_fields, IDType> static_pst,
												IDType id_type)
{
	// Printing function for printing operator << to use, as private data members must be accessed in the process
	{static_pst.print(std::declval<std::ostream &os>())} -> void;

	// Required search functions
	{static_pst.threeSidedSearch(std::declval<size_t &>(/* num_res_elems */),
									std::declval<T>(/* min_dim1_val */),
									std::declval<T>(/* max_dim1_val */),
									std::declval<T>(/* min_dim2_val */))}
								-> PointStructTemplate<T, num_ID_fields, IDType>*;
	{static_pst.twoSidedLeftSearch(std::declval<size_t &>(/* num_res_elems */),
									std::declval<T>(/* max_dim1_val */),
									std::declval<T>(/* min_dim2_val */))}
								-> PointStructTemplate<T, num_ID_fields, IDType>*;
	{static_pst.twoSidedRightSearch(std::declval<size_t &>(/* num_res_elems */),
									std::declval<T>(/* min_dim1_val */),
									std::declval<T>(/* min_dim2_val */))}
								-> PointStructTemplate<T, num_ID_fields, IDType>*;
};

template <class T>
std::ostream &operator<<(std::ostream &os, T &t)
{
	t.print(os);
	return os;
}

#endif
