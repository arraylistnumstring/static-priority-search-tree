#ifndef RAND_DATA_PT_GENERATION_H
#define RAND_DATA_PT_GENERATION_H

#include "pst-test-info-struct.h"

// Pre-condition: num_IDs == 0 <=> id_distr_ptr == nullptr
template <template<typename, typename, size_t> class PointStructTemplate,	
			typename T, typename IDType,
		 	size_t num_IDs, bool vals_inc_ordered,
			template<typename> typename Distrib,
			template<typename> typename IDDistrib,
			typename RandNumEng>
PointStructTemplate<T, IDType, num_IDs> generateRandPts(const size_t num_elems,
														Distrib<T> &val_distr,
														RandNumEng &rand_num_eng,
														Distrib<T> *const inter_size_distr_ptr = nullptr,
														IDDistrib<IDType> *const id_distr_ptr = nullptr)
{
	PointStructTemplate<T, IDType, num_IDs> *pt_arr = new PointStructTemplate<T, IDType, num_IDs>[num_elems];

	for (size_t i = 0; i < num_elems; i++)
	{
		// Distribution takes random number engine as parameter with which to generate its next value
		T val1 = val_distr(rand_num_eng);

		T val2;
		if (inter_size_distr_ptr == nullptr)
			val2 = val_distr(rand_num_eng);
		else
			val2 = val1 + (*inter_size_distr_ptr)(rand_num_eng);

		// Swap generated values only if val1 > val2 and monotonically increasing order is required
		if (vals_inc_ordered && val1 > val2)
		{
			pt_arr[i].dim1_val = val2;
			pt_arr[i].dim2_val = val1;
		}
		else
		{
			pt_arr[i].dim1_val = val1;
			pt_arr[i].dim2_val = val2;
		}
		// Instantiation of value of type IDType
		if constexpr (num_IDs != 0)
			pt_arr[i].id = (*id_distr_ptr)(rand_num_eng);
	}

	return pt_arr;
}

#endif
