#ifndef CLASS_MEMBER_CHECKERS_H
#define CLASS_MEMBER_CHECKERS_H

#include <type_traits>


// Create type trait to determine whether a struct possesses an ID field .id
// Source:
//	https://stackoverflow.com/a/16000226
// Makes use of SFINAE such that HasID inherits from std::false_type in the general case, and inherits from std::true_type if the fully specified version is successfully instantiated for type T
// Returns HasID<T>::value == false
template <typename T, typename = int>
struct HasID: std::false_type {};
// decltype() essentially returns the type of the supplied expression
// Built-in comma operator returns its second argument and throws out the first; use of void casting is valid because any object can be cast to void, and ensures that the built-in comma operator is used, rather than any overrides
// When type T does not have an id field, the attempt to construct the below fully specified HasID struct fails (SFINAE), and the compiler falls back to the general case that inherits from std::false_type
// Use of a default second template type of int ensures that full specification only fails if T::id fails; as this second type is never explicitly used, it does not need to be named
// Returns HasID<T>::value == true
template <typename T>
struct HasID <T, decltype((void) T::id, 0)>: std::true_type {};

#endif
