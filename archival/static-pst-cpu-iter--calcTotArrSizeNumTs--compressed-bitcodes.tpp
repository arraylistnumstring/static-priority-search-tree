// Original calcTotArrSizeNumTs code for compressed-bitcode (multiple bitcodes per byte) solution
template <typename T>
size_t StaticPSTCPUIter<T>::calcTotArrSizeNumTs(const size_t num_elem_slots)
{
	/*
		tot_arr_size_num_Ts = ceil(num_elem_slots * (num_val_subarrs + num_Ts/bitcode))
							= ceil(num_elem_slots * (num_val_subarrs + 1 B/#codes * 1 of T/#Bs))
							= ceil(num_elem_slots * (num_val_subarrs + 1/codes_per_byte * 1/sizeof(T)))
							= ceil(num_elem_slots * num_val_subarrs + num_elem_slots / (codes_per_byte * sizeof(T))
							= num_elem_slots * num_val_subarrs + ceil(num_elem_slots/ (codes_per_byte * sizeof(T))
				if num_elem_slots % codes_per_byte != 0:
							= num_elem_slots * num_val_subarrs + num_elem_slots/ (codes_per_byte * sizeof(T) + 1
				if num_elem_slots % codes_per_byte == 0:
							= num_elem_slots * num_val_subarrs + num_elem_slots/ (codes_per_byte * sizeof(T)
	*/
	size_t tot_arr_size_num_Ts = num_val_subarrs * num_elem_slots + num_elem_slots/(TreeNode::getNumBitcodesPerByte() * sizeof(T));
	if (num_elem_slots % (TreeNode::getNumBitcodesPerByte() * sizeof(T)) != 0)
		tot_arr_size_num_Ts++;
	return tot_arr_size_num_Ts;
}
