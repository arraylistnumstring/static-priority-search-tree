#ifndef TEMPLATE_CLASS_H
#define TEMPLATE_CLASS_H

#include <type_traits>	// To filter out non-numeric types of T

// Allows for insertion of any numeric type T; only compiles if T is numeric (see SFINAE for details)
template
<
	typename T,
	// std::is_arithmetic, std::enable_if: C++11 feature; defined in <type_traits>
	// std::enable_if<condition, T>::type returns T if condition is true, else no such member
	// std::is_arithmetic<T>::value returns true if T is an arithmetic type, false otherwise
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
class TemplateClass
{
	public:
		TemplateClass();
		virtual ~TemplateClass();

		T dim1_val;
		T dim2_val;
};

// Implementation file; for class templates, implementations must be in the same file as the declaration so that the compiler can access them
#include "template-class.tpp"

#endif
