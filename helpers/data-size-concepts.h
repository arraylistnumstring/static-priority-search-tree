#ifndef DATA_SIZE_CONCEPTS_H
#define DATA_SIZE_CONCEPTS_H

template <typename U, typename V>
concept SizeOfUAtLeastSizeOfV = requires (U u, V v) {sizeof(u) >= sizeof(v);};

#endif
