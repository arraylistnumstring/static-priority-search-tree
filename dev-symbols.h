#ifndef DEV_SYMBOLS_H
#define DEV_SYMBOLS_H

// To use a global memory-scoped variable, must declare it outside of any function
// To match a valid atomicAdd function signature, res_arr_ind_d must be declared as an unsigned long long (unsigned long long is the same as an unsigned long long int)
__device__ unsigned long long res_arr_ind_d;

#endif
