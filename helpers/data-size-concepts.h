#ifndef DATA_SIZE_CONCEPTS_H
#define DATA_SIZE_CONCEPTS_H

template <typename U, typename V>
concept SizeOfUAtLeastSizeOfV = requires {sizeof(U) >= sizeof(V);};

template <typename U, typename V>
concept IntSizeOfUAtLeastSizeOfV = SizeOfUAtLeastSizeOfV<U, V>
									&& std::is_integral<U>::value && std::is_integral<V>::value; 

#endif
