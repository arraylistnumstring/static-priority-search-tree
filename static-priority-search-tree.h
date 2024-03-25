#ifndef STATIC_PRIORITY_SEARCH_TREE_H
#define STATIC_PRIORITY_SEARCH_TREE_H

#include <concepts>		// To create abstract class template; requires C++20
#include <iostream>

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			template<typename, template<typename, typename, size_t> class, typename, size_t> class StaticPrioritySearchTree,
			typename IDType=void, size_t num_ID_fields=0
		 >
concept StaticPST = requires (T t, PointStructTemplate<T, IDType, num_ID_fields> ptstr,
								StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_ID_fields> static_pst,
								IDType id_type)
{
	// Printing function for printing operator << to use, as private data members must be accessed in the process
	// {} -> type_func<> passes in result of curly braces to the first template parameter of type_func<>
	{static_pst.print(std::declval<std::ostream &>())} -> std::is_void;

	// Required search functions
	{static_pst.threeSidedSearch(std::declval<size_t &>(/* num_res_elems */),
									std::declval<T>(/* min_dim1_val */),
									std::declval<T>(/* max_dim1_val */),
									std::declval<T>(/* min_dim2_val */))}
								-> std::same_as<PointStructTemplate<T, IDType, num_ID_fields>*>;
	{static_pst.twoSidedLeftSearch(std::declval<size_t &>(/* num_res_elems */),
									std::declval<T>(/* max_dim1_val */),
									std::declval<T>(/* min_dim2_val */))}
								-> std::same_as<PointStructTemplate<T, IDType, num_ID_fields>*>;
	{static_pst.twoSidedRightSearch(std::declval<size_t &>(/* num_res_elems */),
									std::declval<T>(/* min_dim1_val */),
									std::declval<T>(/* min_dim2_val */))}
								-> std::same_as<PointStructTemplate<T, IDType, num_ID_fields>*>;
};

// As the concept StaticPST is not a type, using it as a constraint must be done with a trailing requires keyword
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			template<typename, template<typename, typename, size_t> class, typename, size_t> class StaticPrioritySearchTree,
			typename IDType=void, size_t num_ID_fields=0
		 >
	requires StaticPST<T, PointStructTemplate, StaticPrioritySearchTree, IDType, num_ID_fields>
std::ostream &operator<<(std::ostream &os, StaticPrioritySearchTree<T, PointStructTemplate, IDType, num_ID_fields> &pst)
{
	pst.print(os);
	return os;
}

#endif
