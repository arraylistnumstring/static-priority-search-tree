#ifndef IPS_TESTER_H
#define IPS_TESTER_H

#include <algorithm>	// To use sort()
#include <iostream>
#include <limits>		// To get numeric limits of each datatype
#include <random>		// To use std::mt19937
#include <type_traits>

#include "err-chk.h"
#include "gpu-err-chk.h"
#include "helper-cuda--modified.h"
#include "interval-parallel-search.h"
#include "isosurface-data-processing.h"
#include "preprocessor-symbols.h"
#include "print-array.h"
#include "rand-data-pt-generator.h"


enum DataType {CHAR, DOUBLE, FLOAT, INT, LONG, UNSIGNED_INT, UNSIGNED_LONG};


template <bool timed_CUDA>
struct IPSTester
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

		// Search value to use
		T search_val;

		bool inter_size_distr_active;

		DataTypeWrapper(std::string input_file, T min_val, T max_val,
						T inter_size_val1, T inter_size_val2, T search_val)
			: input_file(input_file),
			rand_num_eng(0),
			distr(min_val, max_val),
			inter_size_distr(inter_size_val2 < 0 ? 0 : inter_size_val1, inter_size_val2 < 0 ? inter_size_val1 : inter_size_val2),
			search_val(search_val),
			inter_size_distr_active(inter_size_val1 >= 0)
		{};

		DataTypeWrapper(std::string input_file, size_t rand_seed, T min_val, T max_val,
						T inter_size_val1, T inter_size_val2, T search_val)
			: input_file(input_file),
			rand_num_eng(rand_seed),
			distr(min_val, max_val),
			inter_size_distr(inter_size_val2 < 0 ? 0 : inter_size_val1, inter_size_val2 < 0 ? inter_size_val1 : inter_size_val2),
			search_val(search_val),
			inter_size_distr_active(inter_size_val1 >= 0)
		{};

		// Nested structs to allow for the metaprogramming equivalent of currying, but with type parameters
		template <template<typename, typename, size_t> class PointStructTemplate, size_t num_IDs>
		struct NumIDsWrapper
		{
			// Nested class have access to all levels of access of their enclosing scope; however, as nested classes are not associated with any enclosing class instance in particular, must keep track of the desired "parent" instance, if any
			DataTypeWrapper<T, Distrib, RandNumEng> ips_tester;

			// Track CUDA device properties; placed in this struct to reduce code redundancy, as this is the only struct that is unspecialised and has a single constructor
			// Data types chosen because they are the ones respectively returned by CUDA
			int num_devs;
			int dev_ind;
			cudaDeviceProp dev_props;

			NumIDsWrapper(DataTypeWrapper<T, Distrib, RandNumEng> ips_tester)
				: ips_tester(ips_tester)
			{
				// Check and save number of GPUs attached to machine
				gpuErrorCheck(cudaGetDeviceCount(&num_devs), "Error in getting number of devices: ");
				if (num_devs < 1)       // No GPUs attached
					throwErr("Error: " + std::to_string(num_devs) + " GPUs attached to host");

				// Use modified version of CUDA's gpuGetMaxGflopsDeviceId() to get top-performing GPU capable of unified virtual addressing; also used so that device in use is the same as that for marching cubes
				dev_ind = gpuGetMaxGflopsDeviceId();
				gpuErrorCheck(cudaGetDeviceProperties(&dev_props, dev_ind),
								"Error in getting device properties of device "
								+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
								+ " total devices: ");

				gpuErrorCheck(cudaSetDevice(dev_ind), "Error setting default device to device "
								+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
								+ " total devices: ");
			};

			template <template <typename> typename IDDistrib, typename IDType>
			struct IDTypeWrapper
			{
				NumIDsWrapper<PointStructTemplate, num_IDs> num_ids_wrapper;

				// Bounds of distribution [a, b] must satisfy b - a <= std::numeric_limits<IDType>::max()
				IDDistrib<IDType> id_distr;

				IDType pt_grid_dims[NUM_DIMS];
				IDType metacell_dims[NUM_DIMS];

				IDTypeWrapper(NumIDsWrapper<PointStructTemplate, num_IDs> num_ids_wrapper)
					: num_ids_wrapper(num_ids_wrapper),
					id_distr(0, std::numeric_limits<IDType>::max())
				{};

				IDTypeWrapper(NumIDsWrapper<PointStructTemplate, num_IDs> num_ids_wrapper,
								IDType pt_grid_dims[NUM_DIMS],
								IDType metacell_dims[NUM_DIMS])
					: num_ids_wrapper(num_ids_wrapper)
				{
					// Attempting to put a const-sized array as part of the member initialiser list causes an error, as the input parameter is treated as being of type IDType *, rather than of type IDType[NUM_DIMS]
					for (int i = 0; i < NUM_DIMS; i++)
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

					void operator()(size_t num_elems, const unsigned num_thread_blocks, const unsigned threads_per_block)
					{
						PointStructTemplate<T, IDType, num_IDs> *pt_arr;

						// Will only be non-nullptr-valued if reading from an input file
						T *vertex_arr = nullptr;

#ifdef DEBUG
						std::cout << "Input file: " << id_type_wrapper.num_ids_wrapper.ips_tester.input_file << '\n';
#endif

						// Place random data generation condition before data input reading so that any timing mechanism for the latter will not be impacted by the evaluation of this conditional
						if (!std::is_integral<IDType>::value ||
								id_type_wrapper.num_ids_wrapper.ips_tester.input_file == "")
						{
							pt_arr = generateRandPts<PointStructTemplate<T, IDType, num_IDs>, T>(num_elems,
														id_type_wrapper.num_ids_wrapper.ips_tester.distr,
														id_type_wrapper.num_ids_wrapper.ips_tester.rand_num_eng,
														true,
														id_type_wrapper.num_ids_wrapper.ips_tester.inter_size_distr_active ? &(id_type_wrapper.num_ids_wrapper.ips_tester.inter_size_distr) : nullptr,
														&(id_type_wrapper.id_distr));

#ifdef DEBUG
							printArray(std::cout, pt_arr, 0, num_elems);
							std::cout << '\n';
#endif
						}
						// Wrap readInVertices in an if constexpr type check (so that it will only be compiled if it succeeds), as pt_grid_dims will be used to allocate memory, and thus must be of integral type
						if constexpr (std::is_integral<IDType>::value)
						{
							if (id_type_wrapper.num_ids_wrapper.ips_tester.input_file != "")
							{
								// Read in vertex array from binary file
								vertex_arr = readInVertices<T>(id_type_wrapper.num_ids_wrapper.ips_tester.input_file,
																id_type_wrapper.pt_grid_dims);

#ifdef DEBUG
								for (IDType k = 0; k < id_type_wrapper.pt_grid_dims[2]; k++)
									for (IDType j = 0; j < id_type_wrapper.pt_grid_dims[1]; j++)
										for (IDType i = 0; i < id_type_wrapper.pt_grid_dims[0]; i++)
										{
											std::cout << 'V';
											std::cout << '[' << i << ']';
											std::cout << '[' << j << ']';
											std::cout << '[' << k << ']';
											std::cout << '=' << vertex_arr[linearVertID(i, j, k, id_type_wrapper.pt_grid_dims)];
											std::cout << '\n';
										}
#endif

								// TODO: Copy vertex_arr to GPU so it's ready for metacell formation and marching cubes

								pt_arr = formMetacells<PointStructTemplate, num_IDs>(vertex_arr,
															id_type_wrapper.pt_grid_dims,
															id_type_wrapper.metacell_dims);
							}
						}

						size_t num_res_elems = 0;
						RetType *res_arr;

						// Variables must be outside of conditionals to be accessible in later conditionals
						cudaEvent_t construct_start, construct_stop, search_start, search_stop;

						if constexpr (timed_CUDA)
						{
							gpuErrorCheck(cudaEventCreate(&construct_start),
											"Error in creating start event for timing CUDA search set-up code");
							gpuErrorCheck(cudaEventCreate(&construct_stop),
											"Error in creating stop event for timing CUDA search set-up code");
							gpuErrorCheck(cudaEventCreate(&search_start),
											"Error in creating start event for timing CUDA search code");
							gpuErrorCheck(cudaEventCreate(&search_stop),
											"Error in creating stop event for timing CUDA search code");

							// Start CUDA search set-up timer (i.e. place this event into default stream)
							gpuErrorCheck(cudaEventRecord(construct_start),
											"Error in recording start event for timing CUDA search set-up code");
						}

						// Allocate space on GPU for input metacell tag array and copy to device
						PointStructTemplate<T, IDType, num_IDs>* pt_arr_d;
						gpuErrorCheck(cudaMalloc(&pt_arr_d, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>)),
										"Error in allocating array to store initial PointStructs on device "
										+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind) + " of "
										+ std::to_string(id_type_wrapper.num_ids_wrapper.num_devs) + ": ");
						gpuErrorCheck(cudaMemcpy(pt_arr_d, pt_arr, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>), cudaMemcpyDefault),
										"Error in copying array of PointStructTemplate<T, IDType, num_IDs> objects to device "
										+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind) + " of "
										+ std::to_string(id_type_wrapper.num_ids_wrapper.num_devs) + ": ");

						if constexpr (timed_CUDA)
						{
							// End CUDA search set-up timer
							gpuErrorCheck(cudaEventRecord(construct_stop),
											"Error in recording stop event for timing CUDA search set-up code");

							// Start CUDA search timer (i.e. place this event in default stream)
							gpuErrorCheck(cudaEventRecord(search_start),
											"Error in recording start event for timing CUDA search code");
						}

						intervalParallelSearch(pt_arr_d, num_elems, res_arr, num_res_elems,
												id_type_wrapper.num_ids_wrapper.ips_tester.search_val,
												id_type_wrapper.num_ids_wrapper.dev_ind,
												id_type_wrapper.num_ids_wrapper.num_devs,
												id_type_wrapper.num_ids_wrapper.dev_props.warpSize,
												num_thread_blocks, threads_per_block);

						if constexpr (timed_CUDA)
						{
							// End CUDA search timer
							gpuErrorCheck(cudaEventRecord(search_stop),
											"Error in recording stop event for timing CUDA search code");

							// Block CPU execution until search stop event has been recorded
							gpuErrorCheck(cudaEventSynchronize(search_stop),
											"Error in blocking CPU execution until completion of stop event for timing CUDA search code");

							// Report construction and search timing
							float ms = 0;	// milliseconds
							gpuErrorCheck(cudaEventElapsedTime(&ms, construct_start, construct_stop),
											"Error in calculating time elapsed for CUDA search set-up code");
							std::cout << "CUDA interval parallel search set-up time: " << ms << " ms\n";

							gpuErrorCheck(cudaEventElapsedTime(&ms, search_start, search_stop),
											"Error in calculating time elapsed for CUDA search code");
							std::cout << "CUDA interval parallel search time: " << ms << " ms\n";

							gpuErrorCheck(cudaEventDestroy(construct_start),
											"Error in destroying start event for timing CUDA search set-up code");
							gpuErrorCheck(cudaEventDestroy(construct_stop),
											"Error in destroying stop event for timing CUDA search set-up code");
							gpuErrorCheck(cudaEventDestroy(search_start),
											"Error in destroying start event for timing CUDA search code");
							gpuErrorCheck(cudaEventDestroy(search_stop),
											"Error in destroying stop event for timing CUDA search code");
						}

						// If result pointer array is on GPU, copy it to CPU and print
						cudaPointerAttributes ptr_info;
						gpuErrorCheck(cudaPointerGetAttributes(&ptr_info, res_arr),
										"Error in determining location type of memory address of result PointStruct array (i.e. whether on host or device)");

						// res_arr is on device; copy to CPU
						if (ptr_info.type == cudaMemoryTypeDevice)
						{
							// Differentiate on-device and on-host pointers
							RetType *res_arr_d = res_arr;

							res_arr = nullptr;

							// Allocate space on host for data
							res_arr = new RetType[num_res_elems];

							if (res_arr == nullptr)
								throwErr("Error: could not allocate RetType array of size "
												+ std::to_string(num_res_elems) + " on host");

							// Copy data from res_arr_d to res_arr
							gpuErrorCheck(cudaMemcpy(res_arr, res_arr_d, num_res_elems * sizeof(RetType),
										cudaMemcpyDefault), 
									"Error in copying array of RetType objects from device "
									+ std::to_string(ptr_info.device) + ": ");

							// Free on-device array of PointStructTemplates
							gpuErrorCheck(cudaFree(res_arr_d), "Error in freeing on-device array of result RetType objects on device "
									+ std::to_string(ptr_info.device) + ": ");
						}

						// Sort output for consistency (specifically compared to GPU-reported outputs, which may be randomly ordered and must therefore be sorted for easy comparisons)
						if constexpr (std::is_same<RetType, IDType>::value)
							std::sort(res_arr, res_arr + num_res_elems,
									[](const IDType &id_1, const IDType &id_2)
									{
										return id_1 < id_2;
									});
						else
							std::sort(res_arr, res_arr + num_res_elems,
									[](const PointStructTemplate<T, IDType, num_IDs> &pt_1,
										const PointStructTemplate<T, IDType, num_IDs> &pt_2)
									{
										return pt_1.compareDim1(pt_2) < 0;
									});

						// For some reason, the offending line is the access of ptstr.print()
						printArray(std::cout, res_arr, 0, num_res_elems);
						std::cout << '\n';

						delete[] pt_arr;
						delete[] res_arr;
					};
				};
			};

			// Template specialisation for case with no ID and therefore no ID distribution; sepcialisation must follow primary (completely unspecified) template; full specialisation not allowed in class scope, hence the remaining dummy type
			template <template <typename> typename IDDistrib>
			struct IDTypeWrapper<IDDistrib, void>
			{
				NumIDsWrapper<PointStructTemplate, num_IDs> num_ids_wrapper;

				IDTypeWrapper(NumIDsWrapper<PointStructTemplate, num_IDs> num_ids_wrapper)
					: num_ids_wrapper(num_ids_wrapper)
				{};

				void operator()(size_t num_elems, const unsigned num_thread_blocks, const unsigned threads_per_block)
				{
					PointStructTemplate<T, void, num_IDs> *pt_arr
						= generateRandPts<PointStructTemplate<T, void, num_IDs>, T, void>(num_elems,
													num_ids_wrapper.ips_tester.distr,
													num_ids_wrapper.ips_tester.rand_num_eng,
													true,
													num_ids_wrapper.ips_tester.inter_size_distr_active ? &(num_ids_wrapper.ips_tester.inter_size_distr) : nullptr);

#ifdef DEBUG
					printArray(std::cout, pt_arr, 0, num_elems);
					std::cout << '\n';
#endif

					size_t num_res_elems = 0;
					PointStructTemplate<T, void, num_IDs> *res_pt_arr;

					// Variables must be outside of conditionals to be accessible in later conditionals
					cudaEvent_t construct_start, construct_stop, search_start, search_stop;

					if constexpr (timed_CUDA)
					{
						gpuErrorCheck(cudaEventCreate(&construct_start),
										"Error in creating start event for timing CUDA search set-up code");
						gpuErrorCheck(cudaEventCreate(&construct_stop),
										"Error in creating stop event for timing CUDA search set-up code");
						gpuErrorCheck(cudaEventCreate(&search_start),
										"Error in creating start event for timing CUDA search code");
						gpuErrorCheck(cudaEventCreate(&search_stop),
										"Error in creating stop event for timing CUDA search code");
						
						// Start CUDA search set-up timer (i.e. place this event into default stream)
						gpuErrorCheck(cudaEventRecord(construct_start),
										"Error in recording start event for timing CUDA search set-up code");
					}

					// Allocate space on GPU for input metacell tag array and copy to device
					PointStructTemplate<T, void, num_IDs>* pt_arr_d;
					gpuErrorCheck(cudaMalloc(&pt_arr_d, num_elems * sizeof(PointStructTemplate<T, void, num_IDs>)),
									"Error in allocating array to store initial PointStructs on device "
									+ std::to_string(num_ids_wrapper.dev_ind) + " of "
									+ std::to_string(num_ids_wrapper.num_devs) + ": ");
					gpuErrorCheck(cudaMemcpy(pt_arr_d, pt_arr, num_elems * sizeof(PointStructTemplate<T, void, num_IDs>), cudaMemcpyDefault),
									"Error in copying array of PointStructTemplate<T, void, num_IDs> objects to device "
									+ std::to_string(num_ids_wrapper.dev_ind) + " of "
									+ std::to_string(num_ids_wrapper.num_devs) + ": ");

					if constexpr (timed_CUDA)
					{
						// End CUDA search set-up timer
						gpuErrorCheck(cudaEventRecord(construct_stop),
										"Error in recording stop event for timing CUDA search set-up code");

						// Start CUDA search timer (i.e. place this event in default stream)
						gpuErrorCheck(cudaEventRecord(search_start),
										"Error in recording start event for timing CUDA search code");
					}

					// Do search and report that returns PointStructTemplate
					intervalParallelSearch(pt_arr_d, num_elems, res_pt_arr, num_res_elems,
											num_ids_wrapper.ips_tester.search_val,
											num_ids_wrapper.dev_ind, num_ids_wrapper.num_devs,
											num_ids_wrapper.dev_props.warpSize,
											num_thread_blocks, threads_per_block);

					if constexpr (timed_CUDA)
					{
						// End CUDA search timer
						gpuErrorCheck(cudaEventRecord(search_stop),
										"Error in recording stop event for timing CUDA search code");

						// Block CPU execution until search stop event has been recorded
						gpuErrorCheck(cudaEventSynchronize(search_stop),
										"Error in blocking CPU execution until completion of stop event for timing CUDA search code");

						// Report construction and search timing
						float ms = 0;	// milliseconds
						gpuErrorCheck(cudaEventElapsedTime(&ms, construct_start, construct_stop),
										"Error in calculating time elapsed for CUDA search set-up code");
						std::cout << "CUDA interval parallel search set-up time: " << ms << " ms\n";

						gpuErrorCheck(cudaEventElapsedTime(&ms, search_start, search_stop),
										"Error in calculating time elapsed for CUDA search code");
						std::cout << "CUDA interval parallel search time: " << ms << " ms\n";

						gpuErrorCheck(cudaEventDestroy(construct_start),
										"Error in destroying start event for timing CUDA search set-up code");
						gpuErrorCheck(cudaEventDestroy(construct_stop),
										"Error in destroying stop event for timing CUDA search set-up code");
						gpuErrorCheck(cudaEventDestroy(search_start),
										"Error in destroying start event for timing CUDA search code");
						gpuErrorCheck(cudaEventDestroy(search_stop),
										"Error in destroying stop event for timing CUDA search code");
					}

					// If result pointer array is on GPU, copy it to CPU and print
					cudaPointerAttributes ptr_info;
					gpuErrorCheck(cudaPointerGetAttributes(&ptr_info, res_pt_arr),
							"Error in determining location type of memory address of result PointStruct array (i.e. whether on host or device)");

					// res_pt_arr is on device; copy to CPU
					if (ptr_info.type == cudaMemoryTypeDevice)
					{
						// Differentiate on-device and on-host pointers
						PointStructTemplate<T, void, num_IDs> *res_pt_arr_d = res_pt_arr;

						res_pt_arr = nullptr;

						// Allocate space on host for data
						res_pt_arr = new PointStructTemplate<T, void, num_IDs>[num_res_elems];

						if (res_pt_arr == nullptr)
							throwErr("Error: could not allocate PointStructTemplate<T, void, num_IDs> array of size "
									+ std::to_string(num_res_elems) + " on host");

						// Copy data from res_pt_arr_d to res_pt_arr
						gpuErrorCheck(cudaMemcpy(res_pt_arr, res_pt_arr_d, num_res_elems * sizeof(PointStructTemplate<T, void, num_IDs>),
									cudaMemcpyDefault), 
								"Error in copying array of PointStructTemplate<T, void, num_IDs> objects from device "
								+ std::to_string(ptr_info.device) + ": ");

						// Free on-device array of PointStructTemplates
						gpuErrorCheck(cudaFree(res_pt_arr_d), "Error in freeing on-device array of result PointStructs on device "
								+ std::to_string(ptr_info.device) + ": ");
					}

					// Sort output for consistency (specifically compared to GPU-reported outputs, which may be randomly ordered and must therefore be sorted for easy comparisons)
					std::sort(res_pt_arr, res_pt_arr + num_res_elems,
							[](const PointStructTemplate<T, void, num_IDs> &pt_1,
								const PointStructTemplate<T, void, num_IDs> &pt_2)
							{
								return pt_1.compareDim1(pt_2) < 0;
							});

					// For some reason, the offending line is the access of ptstr.print()
					printArray(std::cout, res_pt_arr, 0, num_res_elems);
					std::cout << '\n';

					delete[] pt_arr;
					delete[] res_pt_arr;
				};
			};
		};
	};
};

#endif
