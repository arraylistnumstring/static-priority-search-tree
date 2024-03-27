#ifndef STATIC_PST_CONCEPTS_H
#define STATIC_PST_CONCEPTS_H

// Helper concept for calcTotArrSizeNumUs()
template <typename U, typename V>
concept SizeOfUAtLeastSizeOfV = requires (U u, V v) {sizeof(u) >= sizeof(v);};

#endif
