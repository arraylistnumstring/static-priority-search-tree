#ifndef RAND_DATA_PT_GENERATOR_H
#define RAND_DATA_PT_GENERATOR_H

#include "class-member-checkers.h"

// Pre-condition: HasID<PointStruct>::value <=> id_distr_ptr != nullptr
template <class PointStruct, typename T, typename IDDistrib, typename Distrib, typename RandNumEng>
PointStruct *generateRandPts(const size_t num_elems, Distrib &val_distr, RandNumEng &rand_num_eng,
								bool vals_inc_ordered, Distrib *const inter_size_distr_ptr = nullptr,
								IDDistrib *const id_distr_ptr = nullptr)
{
	PointStruct *pt_arr = new PointStruct[num_elems];

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

		// Instantiation of values of type IDType
		if constexpr (HasID<PointStruct>::value)
			pt_arr[i].id = (*id_distr_ptr)(rand_num_eng);
	}

	return pt_arr;
};

#endif
