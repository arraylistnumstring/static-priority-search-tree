#ifndef PST_TESTER_H
#define PST_TESTER_H

#include <limits>		// To get numeric limits of each datatype
#include <random>		// To use std::mt19937
#include <type_traits>

#include "err-chk.h"
#include "gpu-err-chk.h"
#include "helper-cuda--modified.h"
#include "linearise-id.h"					// For NUM_DIMS definition


enum DataType {DOUBLE, FLOAT, INT, LONG, UNSIGNED_INT, UNSIGNED_LONG};

enum PSTTestCodes
{
	CONSTRUCT,
	LEFT_SEARCH,
	RIGHT_SEARCH,
	THREE_SEARCH
};

enum PSTType {CPU_ITER, CPU_RECUR, GPU};

// No integral requirement is imposed on GridDimType, as datasetTest() is still declared as a friend for non-integral IDTypes and would thus result in compilation failure
template <typename PointStruct, typename T, typename StaticPST,
		 	PSTType pst_type, bool timed, typename RetType, typename GridDimType,
			typename PSTTester
		>
void datasetTest(const std::string input_file, const unsigned tree_ops_warps_per_block,
					PSTTester &pst_tester, GridDimType pt_grid_dims[Dims::NUM_DIMS],
					GridDimType metacell_dims[Dims::NUM_DIMS],
					cudaDeviceProp &dev_props, const int num_devs, const int dev_ind);

template <typename PointStruct, typename T, typename IDType, typename StaticPST,
		 	PSTType pst_type, bool timed, typename RetType=PointStruct, typename IDDistribInstan,
			typename PSTTester
		>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStruct>
		>::value
void randDataTest(const size_t num_elems, const unsigned warps_per_block,
					PSTTestCodes test_type, PSTTester &pst_tester,
					cudaDeviceProp &dev_props, const int num_devs, const int dev_ind,
					IDDistribInstan *const id_distr_ptr=nullptr);


template <bool timed>
struct PSTTester
{
	template <typename T,
				template<typename> typename Distrib,
				// Default random number engine: Mersenne Twister 19937; takes its constructor parameter as its seed
				typename RandNumEng=std::mt19937>
		requires std::is_arithmetic<T>::value
	struct DataTypeWrapper
	{
		std::string input_file = "";
		RandNumEng rand_num_eng;
		Distrib<T> distr;
		Distrib<T> inter_size_distr;

		// Search values to use; may not be used if test is not of search type
		T dim1_val_bound1;
		T dim1_val_bound2;
		T min_dim2_val;

		// Only queried when generating random data, but set consistently with the operation being done in all cases for ease of debugging
		bool vals_inc_ordered;
		bool inter_size_distr_active;

		DataTypeWrapper(std::string input_file, T min_val, T max_val,
						T inter_size_val1, T inter_size_val2,
						T dim1_val_bound1, T dim1_val_bound2,
						T min_dim2_val, bool vals_inc_ordered)
			: input_file(input_file),
			rand_num_eng(0),
			distr(min_val, max_val),
			inter_size_distr(inter_size_val2 < 0 ? 0 : inter_size_val1, inter_size_val2 < 0 ? inter_size_val1 : inter_size_val2),
			dim1_val_bound1(dim1_val_bound1),
			dim1_val_bound2(dim1_val_bound2),
			min_dim2_val(min_dim2_val),
			vals_inc_ordered(vals_inc_ordered),
			inter_size_distr_active(inter_size_val1 >= 0)
		{};

		DataTypeWrapper(std::string input_file, size_t rand_seed, T min_val, T max_val,
						T inter_size_val1, T inter_size_val2, T dim1_val_bound1,
						T dim1_val_bound2, T min_dim2_val, bool vals_inc_ordered)
			: input_file(input_file),
			rand_num_eng(rand_seed),
			distr(min_val, max_val),
			inter_size_distr(inter_size_val2 < 0 ? 0 : inter_size_val1, inter_size_val2 < 0 ? inter_size_val1 : inter_size_val2),
			dim1_val_bound1(dim1_val_bound1),
			dim1_val_bound2(dim1_val_bound2),
			min_dim2_val(min_dim2_val),
			vals_inc_ordered(vals_inc_ordered),
			inter_size_distr_active(inter_size_val1 >= 0)
		{};

		// Nested structs to allow for the metaprogramming equivalent of currying, but with type parameters
		template <template<typename, typename, size_t> class PointStructTemplate,
					template<typename, template<typename, typename, size_t> class, typename, size_t> class StaticPSTTemplate,
					PSTType pst_type>
		struct TreeTypeWrapper
		{
			// Nested class have access to all levels of access of their enclosing scope; however, as nested classes are not associated with any enclosing class instance in particular, so must keep track of the desired "parent" instance, if any
			DataTypeWrapper<T, Distrib, RandNumEng> pst_tester;

			TreeTypeWrapper(DataTypeWrapper<T, Distrib, RandNumEng> pst_tester)
				: pst_tester(pst_tester)
			{};

			template <size_t num_IDs>
			struct NumIDsWrapper
			{
				TreeTypeWrapper<PointStructTemplate, StaticPSTTemplate, pst_type> tree_type_wrapper;
				
				// Track CUDA device properties; placed in this struct to reduce code redundancy, as this is the only struct that is unspecialised and has a single constructor
				// Data types chosen because they are the ones respectively returned by CUDA
				int num_devs;
				int dev_ind;
				cudaDeviceProp dev_props;

				NumIDsWrapper(TreeTypeWrapper<PointStructTemplate, StaticPSTTemplate, pst_type> tree_type_wrapper)
					: tree_type_wrapper(tree_type_wrapper)
				{
					// Check and save number of GPUs attached to machine
					gpuErrorCheck(cudaGetDeviceCount(&num_devs), "Error in getting number of devices: ");
					if (num_devs < 1)       // No GPUs attached
						throwErr("Error: " + std::to_string(num_devs) + " GPUs attached to host");

					// Use modified version of CUDA's gpuGetMaxGflopsDeviceId() to get top-performing GPU capable of unified virtual addressing; also used so that device in use is the same as that for marching cubes
					dev_ind = gpuGetMaxGflopsDeviceId();
					gpuErrorCheck(cudaGetDeviceProperties(&dev_props, dev_ind),
									"Error in getting device properties of device "
									+ std::to_string(dev_ind + 1) + " (1-indexed) of "
									+ std::to_string(num_devs) + ": "
								);

					gpuErrorCheck(cudaSetDevice(dev_ind), "Error setting default device to device "
									+ std::to_string(dev_ind + 1) + " (1-indexed) of "
									+ std::to_string(num_devs) + ": "
								);
				};

				template <template<typename> typename IDDistrib, typename IDType>
				struct IDTypeWrapper
				{
					NumIDsWrapper<num_IDs> num_ids_wrapper;

					// Bounds of distribution [a, b] must satisfy b - a <= std::numeric_limits<IDType>::max()
					IDDistrib<IDType> id_distr;

					IDType pt_grid_dims[Dims::NUM_DIMS];
					IDType metacell_dims[Dims::NUM_DIMS];

					IDTypeWrapper(NumIDsWrapper<num_IDs> num_ids_wrapper)
						: num_ids_wrapper(num_ids_wrapper),
						id_distr(0, std::numeric_limits<IDType>::max())
					{};

					IDTypeWrapper(NumIDsWrapper<num_IDs> num_ids_wrapper,
									IDType pt_grid_dims[Dims::NUM_DIMS],
									IDType metacell_dims[Dims::NUM_DIMS])
						: num_ids_wrapper(num_ids_wrapper)
					{
						// Attempting to put a const-sized array as part of the member initialiser list causes an error, as the input parameter is treated as being of type IDType *, rather than of type IDType[Dims::NUM_DIMS]
						for (int i = 0; i < Dims::NUM_DIMS; i++)
						{
							this->pt_grid_dims[i] = pt_grid_dims[i];
							this->metacell_dims[i] = metacell_dims[i];
						}
					};

					template <typename RetType=PointStructTemplate<T, IDType, num_IDs>>
						// Requires that RetType is either of type IDType or of type PointStructTemplate<T, IDType, num_IDs>
						// std::disjunction<B1, ..., Bn> performs a logical OR on enclosed type traits, specifically on the value returned by \Vee_{i=1}^n bool(B1::value)
						requires std::disjunction<
											std::is_same<RetType, IDType>,
											std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
							>::value
					struct RetTypeWrapper
					{
						IDTypeWrapper<IDDistrib, IDType> id_type_wrapper;

						RetTypeWrapper(IDTypeWrapper<IDDistrib, IDType> id_type_wrapper)
							: id_type_wrapper(id_type_wrapper)
						{};

						void operator()(size_t num_elems, const unsigned warps_per_block, PSTTestCodes test_type=CONSTRUCT)
						{
							// To avoid instantiating a float-type GridDimType in datasetTest(), construct the two complimentary conditions explicitly (with one being a constexpr)
							if constexpr (std::is_integral<IDType>::value)
							{
								if (id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.input_file != "")
								{
									datasetTest<PointStructTemplate<T, IDType, num_IDs>, T,
													StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>,
													pst_type, timed, RetType
												>
													(id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.input_file,
														warps_per_block,
														id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester,
														id_type_wrapper.pt_grid_dims,
														id_type_wrapper.metacell_dims,
														id_type_wrapper.num_ids_wrapper.dev_props,
														id_type_wrapper.num_ids_wrapper.num_devs,
														id_type_wrapper.num_ids_wrapper.dev_ind
													);
								}
							}
							if (!std::is_integral<IDType>::value
									|| id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.input_file == "")
							{
								randDataTest<PointStructTemplate<T, IDType, num_IDs>, T, IDType,
												StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>,
												pst_type, timed, RetType
											>
												(num_elems, warps_per_block, test_type,
												 id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester,
												 id_type_wrapper.num_ids_wrapper.dev_props,
												 id_type_wrapper.num_ids_wrapper.num_devs,
												 id_type_wrapper.num_ids_wrapper.dev_ind,
												 &(id_type_wrapper.id_distr)
												);
							}
						};

						// Declare a particular full specification of the test functions as friends to this struct; requires a declaration of the template function before this use as well
						friend void datasetTest<PointStructTemplate<T, IDType, num_IDs>, T,
													StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>,
													pst_type, timed, RetType
												>
													(const std::string input_file, const unsigned tree_ops_warps_per_block,
														DataTypeWrapper<T, Distrib, RandNumEng> &pst_tester,
														IDType pt_grid_dims[Dims::NUM_DIMS],
														IDType metacell_dims[Dims::NUM_DIMS],
														cudaDeviceProp &dev_props, const int num_devs,
														const int dev_ind
													);

						friend void randDataTest<PointStructTemplate<T, IDType, num_IDs>, T, IDType,
							   						StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>,
													pst_type, timed, RetType
												>
													(const size_t num_elems,
													 	const unsigned warps_per_block,
													 	PSTTestCodes test_type,
														DataTypeWrapper<T, Distrib, RandNumEng> &pst_tester,
														cudaDeviceProp &dev_props,
														const int num_devs, const int dev_ind,
														IDDistrib<IDType> *const id_distr_ptr
													);
					};
				};

				// Template specialisation for case with no ID and therefore no ID distribution; sepcialisation must follow primary (completely unspecified) template; full specialisation not allowed in class scope, hence the remaining dummy type
				template <template<typename> typename IDDistrib>
				struct IDTypeWrapper<IDDistrib, void>
				{
					NumIDsWrapper<num_IDs> num_ids_wrapper;

					IDTypeWrapper(NumIDsWrapper<num_IDs> num_ids_wrapper)
						: num_ids_wrapper(num_ids_wrapper)
					{};

					void operator()(size_t num_elems, const unsigned warps_per_block, PSTTestCodes test_type=CONSTRUCT)
					{
						// Points without IDs can only run randDataTest()
						randDataTest<PointStructTemplate<T, void, num_IDs>, T, void,
										StaticPSTTemplate<T, PointStructTemplate, void, num_IDs>,
										pst_type, timed,
										PointStructTemplate<T, void, num_IDs>,
										IDDistrib<void>
									>
										(num_elems, warps_per_block, test_type,
											num_ids_wrapper.tree_type_wrapper.pst_tester,
											num_ids_wrapper.dev_props,
											num_ids_wrapper.num_devs, num_ids_wrapper.dev_ind);
					};

					friend void randDataTest<PointStructTemplate<T, void, num_IDs>, T, void,
						   						StaticPSTTemplate<T, PointStructTemplate, void, num_IDs>,
												pst_type, timed
											>
												(const size_t num_elems, const unsigned warps_per_block,
													PSTTestCodes test_type,
													DataTypeWrapper<T, Distrib, RandNumEng> &pst_tester,
													cudaDeviceProp &dev_props,
													const int num_devs, const int dev_ind,
													IDDistrib<void> *const id_distr_ptr
												);
				};
			};
		};
	};
};

#include "pst-tester-functions.tu"

#endif
