#ifndef TYPE_CONCEPTS_H
#define TYPE_CONCEPTS_H

#include <type_traits>

template <typename T>
concept NonVoidType = !std::is_void<T>::value;

template <typename U, typename V>
concept SizeOfUAtLeastSizeOfV = requires {sizeof(U) >= sizeof(V);};

template <typename U, typename V>
concept IntSizeOfUAtLeastSizeOfV = SizeOfUAtLeastSizeOfV<U, V>
									&& std::is_integral<U>::value && std::is_integral<V>::value; 

#endif
