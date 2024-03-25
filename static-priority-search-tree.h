#ifndef STATIC_PRIORITY_SEARCH_TREE_H
#define STATIC_PRIORITY_SEARCH_TREE_H

#include <concepts>
#include <iostream>
#include <type_traits>	// To filter out non-numeric types of T

// Use concepts, a C++20 feature, to determine that the provided PtStruct type is valid
// PtStruct is a template template parameter, i.e. is a template parameter that takes in a template parameter, so has template in front of its definition
template <typename T, template <typename, size_t=0, typename=void> class PtStructTemplate>
concept ValidPtStruct = requires(T t, PtStructTemplate<T> ptstr)
{
	// Throws a compile-time error if T is not of arithmetic (numeric) type
	requires std::is_arithmetic<T>::value;
	// Checks for the existence of the following data members and member functions
	ptstr.dim1_val;
	ptstr.dim2_val;
	ptstr.compareDim1(ptstr);
	ptstr.compareDim2(ptstr);
};

template <typename T, template <typename, size_t, typename> class PtStructIDTemplate,
			size_t num_ID_fields, typename IDType>
concept ValidPtStructID = requires (T t, PtStructIDTemplate<T, num_ID_fields, IDType> ptstr_id, IDType id_type)
{
	requires ValidPtStruct<T, PtStructIDTemplate>;
	requires num_ID_fields > 0;
	requires !std::is_void<IDType>::value;
	ptstr_id.ids;
};

template <typename T, template<typename, size_t, typename> class PtStructTemplate,
			template<typename, template<typename, size_t, typename> class, size_t, typename> class StaticPrioritySearchTree,
			size_t num_ID_fields=0, typename IDType=void>
// Functionally an abstract class or an interface
concept StaticPrioritySearchTree = requires (T t, PtStructTemplate<T, num_ID_fields, IDType> ptstr,
												StaticPrioritySearchTree<T, PtStructTemplate, num_ID_fields, IDType> static_pst,
												IDType id_type)
{
	requires ((ValidPtStruct<T, PtStructTemplate> PtStruct)
	{
		requires num_ID_fields == 0;
		{static_pst.threeSidedSearch(size_t &num_res_elems, T min_dim1_val, T max_dim1_val, T min_dim2_val)} -> PtStruct*;
		{static_pst.twoSidedLeftSearch(size_t &num_res_elems, T max_dim1_val, T min_dim2_val)} -> PtStruct*;
		{static_pst.twoSidedRightSearch(size_t &num_res_elems, T min_dim1_val, T min_dim2_val)} -> PtStruct*;
	})

			|| (ValidPtStructID<T, PtStructTemplate, num_ID_fields, IDType> PtStructID)
	{
		{static_pst.threeSidedSearch(size_t &num_res_elems, T min_dim1_val, T max_dim1_val, T min_dim2_val)} -> PtStructID*;
		{static_pst.twoSidedLeftSearch(size_t &num_res_elems, T max_dim1_val, T min_dim2_val)} -> PtStructID*;
		{static_pst.twoSidedRightSearch(size_t &num_res_elems, T min_dim1_val, T min_dim2_val)} -> PtStructID*;
	};
	requires (std::ostream &os) {
		// Printing function for printing operator << to use, as private data members must be accessed in the process
		{static_pst.print(os)} -> void;
	};
};

#endif
