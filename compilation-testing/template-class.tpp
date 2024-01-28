#ifndef TEMPLATE_CLASS_TPP
#define TEMPLATE_CLASS_TPP

#ifndef TEMPLATE_CLASS_H
#error __FILE__ should only be included from template-class.h.
#endif

// Because a second typename was used to ensure that T is a numeric type, this second typename must be named in functions defined outside of the class declaration; also, parameters and return types must still only use the first type; see
// https://cplusplus.com/forum/beginner/267953/
template <typename T, typename U>
TemplateClass<T, U>::TemplateClass()
	: dim1_val(0),
	dim2_val(0)
{}

template <typename T, typename U>
TemplateClass<T, U>::~TemplateClass()
{}

#endif
