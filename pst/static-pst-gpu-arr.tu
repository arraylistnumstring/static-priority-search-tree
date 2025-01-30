template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::StaticPSTGPUArr(PointStructTemplate<T, IDType, num_IDs> *const &pt_arr_d,
																			size_t num_elems,
																			const int warps_per_block,
																			int dev_ind, int num_devs,
																			cudaDeviceProp dev_props)
	: num_elems(num_elems),
	warps_per_block(warps_per_block),
	dev_ind(dev_ind),
	num_devs(num_devs),
	dev_props(dev_props)
{
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTGPUArr<T, PointStructTemplate, IDType, num_IDs>::print(std::ostream &os) const
{
}
