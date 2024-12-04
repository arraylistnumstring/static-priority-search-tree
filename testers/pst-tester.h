#ifndef PST_TESTER_H
#define PST_TESTER_H

#include <algorithm>	// To use sort()
#include <chrono>		// To use std::chrono::steady_clock (a monotonic clock suitable for interval measurements) and related functions
#include <ctime>		// To use std::clock_t (CPU timer; pauses when CPU pauses, etc.)
#include <iostream>
#include <limits>		// To get numeric limits of each datatype
#include <random>		// To use std::mt19937
#include <type_traits>

#include "data-size-concepts.h"
#include "err-chk.h"
#include "gpu-err-chk.h"
#include "helper-cuda--modified.h"
#include "isosurface-data-processing.h"
#include "linearise-id.h"					// For NUM_DIMS definition
#include "print-array.h"
#include "rand-data-pt-generator.h"


enum DataType {DOUBLE, FLOAT, INT, LONG, UNSIGNED_INT, UNSIGNED_LONG};

enum PSTTestCodes
{
	CONSTRUCT,
	LEFT_SEARCH,
	RIGHT_SEARCH,
	THREE_SEARCH
};

enum PSTType {CPU_ITER, CPU_RECUR, GPU};


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
									+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
									+ " total devices: ");

					gpuErrorCheck(cudaSetDevice(dev_ind), "Error setting default device to device "
									+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
									+ " total devices: ");
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
							PointStructTemplate<T, IDType, num_IDs> *pt_arr;

							// Will only be non-nullptr-valued if reading from an input file
							T *vertex_arr_d = nullptr;

							// Variables must be outside of conditionals to be accessible in later conditionals
							// Place CPU timing here, as GPU -> CPU transfer time of metacells must be taken into consideration for the "construct" cost
							std::clock_t construct_start_CPU, construct_stop_CPU, search_start_CPU, search_stop_CPU;
							std::chrono::time_point<std::chrono::steady_clock> construct_start_wall, construct_stop_wall, search_start_wall, search_stop_wall;

#ifdef DEBUG
							std::cout << "Input file: " << id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.input_file << '\n';
#endif

							// Place random data generation condition before data input reading so that any timing mechanism for the latter will not be impacted by the evaluation of this conditional
							if (!std::is_integral<IDType>::value ||
									id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.input_file == "")
							{
								pt_arr = generateRandPts<PointStructTemplate<T, IDType, num_IDs>, T>(num_elems,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.distr,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.rand_num_eng,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.vals_inc_ordered,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.inter_size_distr_active ? &(id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.inter_size_distr) : nullptr,
															&(id_type_wrapper.id_distr));

#ifdef DEBUG
								printArray(std::cout, pt_arr, 0, num_elems);
								std::cout << '\n';
#endif
							}
							// Wrap readInVertices in an if constexpr type check (so that it will only be compiled if it succeeds), as pt_grid_dims will be used to allocate memory, and thus must be of integral type
							if constexpr (std::is_integral<IDType>::value)
							{
								if (id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.input_file != "")
								{
									IDType num_verts = 1;
									for (int i = 0; i < Dims::NUM_DIMS; i++)
										num_verts *= id_type_wrapper.pt_grid_dims[i];

									// Read in vertex array from binary file
									T *vertex_arr = readInVertices<T>(id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.input_file,
																			num_verts
																		);

#ifdef DEBUG
									// Instantiate as separate variable, as attempting a direct substitution of an array initialiser doesn't compile, even if statically cast to an appropriate type
									IDType start_inds[Dims::NUM_DIMS] = {0, 0, 0};
									print3DArray(std::cout, vertex_arr, start_inds,
													id_type_wrapper.pt_grid_dims);
#endif

									// Copy vertex_arr to GPU so it's ready for metacell formation and marching cubes
									gpuErrorCheck(cudaMalloc(&vertex_arr_d, num_verts * sizeof(T)),
														"Error in allocating vertex storage array on device "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind)
														+ " of "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.num_devs)
														+ " total devices: "
													);
									// Implicitly synchronous from the host's point of view as only pinned memory can have truly asynchronous cudaMemcpy() calls
									gpuErrorCheck(cudaMemcpy(vertex_arr_d, vertex_arr, num_verts * sizeof(T), cudaMemcpyDefault),
														"Error in copying array of vertices to device "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind)
														+ " of "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.num_devs)
														+ " total devices: "
													);

									// Prior cudaMemcpy() is staged, if not already written through, so can free vertex_arr
									delete[] vertex_arr;

									pt_arr = formMetacells<PointStructTemplate<T, IDType, num_IDs>>
																(
																	vertex_arr_d,
																	id_type_wrapper.pt_grid_dims,
																	id_type_wrapper.metacell_dims,
																	num_elems,
																	id_type_wrapper.num_ids_wrapper.dev_ind,
																	id_type_wrapper.num_ids_wrapper.num_devs,
																	id_type_wrapper.num_ids_wrapper.dev_props.warpSize
																);

								}
							}

							if constexpr (pst_type != GPU)
							{
								// CPU PST-only construction cost of copying data to host must be timed
								if constexpr (timed)
								{
									construct_start_CPU = std::clock();
									construct_start_wall = std::chrono::steady_clock::now();
								}
								if constexpr (std::is_integral<IDType>::value)
								{
									if (vertex_arr_d != nullptr)
									{
										// Copy metacell array to CPU for iterative and recursive PSTs to process
										PointStructTemplate<T, IDType, num_IDs> *pt_arr_host
											= new PointStructTemplate<T, IDType, num_IDs>[num_elems];
										gpuErrorCheck(cudaMemcpy(pt_arr_host, pt_arr, num_elems * sizeof(PointStructTemplate<T, IDType, num_IDs>), cudaMemcpyDefault),
														"Error in copying on-device array of metacell PointStructs to host from device "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind)
														+ " of "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.num_devs)
														+ " total devices: ");
										gpuErrorCheck(cudaFree(pt_arr),
														"Error in freeing on-device array of metacell PointStructs on device "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind)
														+ " of "
														+ std::to_string(id_type_wrapper.num_ids_wrapper.num_devs)
														+ " total devices: ");

										pt_arr = pt_arr_host;
									}
								}
							}

							// Check that GPU memory is sufficiently big for the necessary calculations
							if constexpr (pst_type == GPU)
							{
								const size_t global_mem_needed = StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>::calcGlobalMemNeeded(num_elems);

								if (global_mem_needed > id_type_wrapper.num_ids_wrapper.dev_props.totalGlobalMem)
								{
									throwErr("Error: needed global memory space of "
												+ std::to_string(global_mem_needed)
												+ " B required for data structure and processing exceeds limit of global memory = "
												+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_props.totalGlobalMem)
												+ " B on device "
												+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind)
												+ " of "
												+ std::to_string(id_type_wrapper.num_ids_wrapper.num_devs)
												+ " total devices: ");
								}
							}


							// Set GPU pending kernel queue size limit; note that the queue takes up global memory, hence why a kernel launch that exceeds the queue's capacity may cause an "Invalid __global__ write of n bytes" error message in compute-sanitizer that points to the line of one of its function parameters
							if constexpr (pst_type == GPU)
							{
								if (num_elems/2 > 2048)		// Default value: 2048
									gpuErrorCheck(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, num_elems/2),
													"Error in increasing pending kernel queue size to "
													+ std::to_string(num_elems/2) + " on device "
													+ std::to_string(id_type_wrapper.num_ids_wrapper.dev_ind)
													+ " of " + std::to_string(id_type_wrapper.num_ids_wrapper.num_devs)
													+ " total devices: ");
							}

							cudaEvent_t construct_start_CUDA, construct_stop_CUDA, search_start_CUDA, search_stop_CUDA;

							// Construction has already started for CPU (because it includes copying of metacells to host), so just start timing GPU construction here
							if constexpr (timed && pst_type == GPU)
							{
								gpuErrorCheck(cudaEventCreate(&construct_start_CUDA),
												"Error in creating start event for timing CUDA PST construction code");
								gpuErrorCheck(cudaEventCreate(&construct_stop_CUDA),
												"Error in creating stop event for timing CUDA PST construction code");
								// For accuracy of measurement of search speeds, create and place search events in stream, even if this is a construction-only test
								gpuErrorCheck(cudaEventCreate(&search_start_CUDA),
												"Error in creating start event for timing CUDA search code");
								gpuErrorCheck(cudaEventCreate(&search_stop_CUDA),
												"Error in creating stop event for timing CUDA search code");

								// Start CUDA construction timer (i.e. place this event into default stream)
								gpuErrorCheck(cudaEventRecord(construct_start_CUDA),
											"Error in recording start event for timing CUDA PST construction code");
							}

							StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs> *tree;
							if constexpr (pst_type == GPU)
								tree = new StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>(pt_arr, num_elems, warps_per_block,
											id_type_wrapper.num_ids_wrapper.dev_ind,
											id_type_wrapper.num_ids_wrapper.num_devs,
											id_type_wrapper.num_ids_wrapper.dev_props);
							else
								tree = new StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>(pt_arr, num_elems);

							if constexpr (timed)
							{
								if constexpr (pst_type == GPU)
								{
									// End CUDA construction timer
									gpuErrorCheck(cudaEventRecord(construct_stop_CUDA),
													"Error in recording stop event for timing CUDA PST construction code");
								}
								else
								{
									construct_stop_CPU = std::clock();
									construct_stop_wall = std::chrono::steady_clock::now();
								}
							}

							if (tree == nullptr)
							{
								throwErr("Error: Could not allocate memory for priority search tree");
								return;
							}

							std::cout << *tree << '\n';

							size_t num_res_elems = 0;
							RetType *res_arr;

							if constexpr (timed)
							{
								if constexpr (pst_type == GPU)
								{
									// Start CUDA search timer (i.e. place this event in default stream)
									gpuErrorCheck(cudaEventRecord(search_start_CUDA),
													"Error in recording start event for timing CUDA search code");
								}
								else
								{
									search_start_CPU = std::clock();
									search_start_wall = std::chrono::steady_clock::now();
								}
							}

							// Search/report test phase
							if constexpr (pst_type == GPU)
							{
								if (test_type == LEFT_SEARCH)
								{
									tree->twoSidedLeftSearch(num_res_elems, res_arr,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val,
																warps_per_block);
								}
								else if (test_type == RIGHT_SEARCH)
								{
									tree->twoSidedRightSearch(num_res_elems, res_arr,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val,
																warps_per_block);
								}
								else if (test_type == THREE_SEARCH)
								{
									tree->threeSidedSearch(num_res_elems, res_arr,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound2,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val,
															warps_per_block);
								}
							}
							else
							{
								if (test_type == LEFT_SEARCH)
								{
									tree->twoSidedLeftSearch(num_res_elems, res_arr,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
								}
								else if (test_type == RIGHT_SEARCH)
								{
									tree->twoSidedRightSearch(num_res_elems, res_arr,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
																id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
								}
								else if (test_type == THREE_SEARCH)
								{
									tree->threeSidedSearch(num_res_elems, res_arr,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound2,
															id_type_wrapper.num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
								}
							}
							// If test_type == CONSTRUCT, do nothing for the search/report phase
							
							if constexpr (timed)
							{
								if constexpr (pst_type == GPU)
								{
									// End CUDA search timer
									gpuErrorCheck(cudaEventRecord(search_stop_CUDA),
													"Error in recording stop event for timing CUDA search code");

									// Block CPU execution until search stop event has been recorded
									gpuErrorCheck(cudaEventSynchronize(search_stop_CUDA),
												"Error in blocking CPU execution until completion of stop event for timing CUDA search code");

									// Report construction and search timing
									// Type chosen because of type of parameter of cudaEventElapsedTime
									float ms = 0;	// milliseconds

									gpuErrorCheck(cudaEventElapsedTime(&ms, construct_start_CUDA, construct_stop_CUDA),
													"Error in calculating time elapsed for CUDA PST construction code");
									std::cout << "CUDA PST construction time: " << ms << " ms\n";

									if (test_type != CONSTRUCT)
									{
										gpuErrorCheck(cudaEventElapsedTime(&ms, search_start_CUDA, search_stop_CUDA),
														"Error in calculating time elapsed for CUDA search code");
										std::cout << "CUDA PST search time: " << ms << " ms\n";
									}

									gpuErrorCheck(cudaEventDestroy(construct_start_CUDA),
													"Error in destroying start event for timing CUDA PST construction code");
									gpuErrorCheck(cudaEventDestroy(construct_stop_CUDA),
													"Error in destroying stop event for timing CUDA PST construction code");
									gpuErrorCheck(cudaEventDestroy(search_start_CUDA),
													"Error in destroying start event for timing CUDA search code");
									gpuErrorCheck(cudaEventDestroy(search_stop_CUDA),
													"Error in destroying stop event for timing CUDA search code");
								}
								else
								{
									search_stop_CPU = std::clock();
									search_stop_wall = std::chrono::steady_clock::now();

									std::cout << "CPU PST construction time:\n"
											  << "\tCPU clock time used:\t"
											  << 1000.0 * (construct_stop_CPU - construct_start_CPU) / CLOCKS_PER_SEC << " ms\n"
											  << "\tWall clock time passed:\t"
											  << std::chrono::duration<double, std::milli>(construct_stop_wall - construct_start_wall).count()
											  << " ms\n";


									if (test_type != CONSTRUCT)
									{
										std::cout << "CPU PST search time:\n"
												  << "\tCPU clock time used:\t"
												  << 1000.0 * (search_stop_CPU - search_start_CPU) / CLOCKS_PER_SEC << " ms\n"
												  << "\tWall clock time passed:\t"
												  << std::chrono::duration<double, std::milli>(search_stop_wall - search_start_wall).count()
												  << " ms\n";
									}
								}
							}

							// If result pointer array is on GPU, copy it to CPU and print
							if constexpr (pst_type == GPU)
							{
								cudaPointerAttributes ptr_info;
								gpuErrorCheck(cudaPointerGetAttributes(&ptr_info, res_arr),
												"Error in determining location type of memory address of result RetType array (i.e. whether on host or device)");

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

							delete tree;
							delete[] pt_arr;
							delete[] res_arr;
						};
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
						PointStructTemplate<T, void, num_IDs> *pt_arr
							= generateRandPts<PointStructTemplate<T, void, num_IDs>, T, void>(num_elems,
												num_ids_wrapper.tree_type_wrapper.pst_tester.distr,
												num_ids_wrapper.tree_type_wrapper.pst_tester.rand_num_eng,
												num_ids_wrapper.tree_type_wrapper.pst_tester.vals_inc_ordered,
												num_ids_wrapper.tree_type_wrapper.pst_tester.inter_size_distr_active ? &(num_ids_wrapper.tree_type_wrapper.pst_tester.inter_size_distr) : nullptr);

#ifdef DEBUG
						printArray(std::cout, pt_arr, 0, num_elems);
						std::cout << '\n';
#endif

						// Check that GPU memory is sufficiently big for the necessary calculations
						if constexpr (pst_type == GPU)
						{
							const size_t global_mem_needed = StaticPSTTemplate<T, PointStructTemplate, void, num_IDs>::calcGlobalMemNeeded(num_elems);

							if (global_mem_needed > num_ids_wrapper.dev_props.totalGlobalMem)
							{
								throwErr("Error: needed global memory space of "
											+ std::to_string(global_mem_needed)
											+ " B required for data structure and processing exceeds limit of global memory = "
											+ std::to_string(num_ids_wrapper.dev_props.totalGlobalMem)
											+ " B on device " + std::to_string(num_ids_wrapper.dev_ind)
											+ " of " + std::to_string(num_ids_wrapper.num_devs)
											+ " total devices: ");
							}
						}

						// Set GPU pending kernel queue size limit; note that the queue takes up global memory, hence why a kernel launch that exceeds the queue's capacity may cause an "Invalid __global__ write of n bytes" error message in compute-sanitizer that points to the line of one of its function parameters
						if constexpr (pst_type == GPU)
						{
							if (num_elems/2 > 2048)		// Default value: 2048
								gpuErrorCheck(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, num_elems/2),
												"Error in increasing pending kernel queue size to "
												+ std::to_string(num_elems/2) + " on device "
												+ std::to_string(num_ids_wrapper.dev_ind)
												+ " of " + std::to_string(num_ids_wrapper.num_devs)
												+ " total devices: ");
						}


						// Variables must be outside of conditionals to be accessible in later conditionals
						cudaEvent_t construct_start_CUDA, construct_stop_CUDA, search_start_CUDA, search_stop_CUDA;
						std::clock_t construct_start_CPU, construct_stop_CPU, search_start_CPU, search_stop_CPU;
						std::chrono::time_point<std::chrono::steady_clock> construct_start_wall, construct_stop_wall, search_start_wall, search_stop_wall;

						if constexpr (timed)
						{
							if constexpr (pst_type == GPU)
							{
								gpuErrorCheck(cudaEventCreate(&construct_start_CUDA),
												"Error in creating start event for timing CUDA PST construction code");
								gpuErrorCheck(cudaEventCreate(&construct_stop_CUDA),
												"Error in creating stop event for timing CUDA PST construction code");
								// For accuracy of measurement of search speeds, create and place search events in stream, even if this is a construction-only test
								gpuErrorCheck(cudaEventCreate(&search_start_CUDA),
												"Error in creating start event for timing CUDA search code");
								gpuErrorCheck(cudaEventCreate(&search_stop_CUDA),
												"Error in creating stop event for timing CUDA search code");

								// Start CUDA construction timer (i.e. place this event into default stream)
								gpuErrorCheck(cudaEventRecord(construct_start_CUDA),
											"Error in recording start event for timing CUDA PST construction code");
							}
							else
							{
								construct_start_CPU = std::clock();
								construct_start_wall = std::chrono::steady_clock::now();
							}
						}

						StaticPSTTemplate<T, PointStructTemplate, void, num_IDs> *tree;
						if constexpr (pst_type == GPU)
							tree = new StaticPSTTemplate<T, PointStructTemplate, void, num_IDs>(pt_arr, num_elems, warps_per_block,
											num_ids_wrapper.dev_ind,
											num_ids_wrapper.num_devs,
											num_ids_wrapper.dev_props);
						else
							tree = new StaticPSTTemplate<T, PointStructTemplate, void, num_IDs>(pt_arr, num_elems);

						if constexpr (timed)
						{
							if constexpr (pst_type == GPU)
							{
								// End CUDA construction timer
								gpuErrorCheck(cudaEventRecord(construct_stop_CUDA),
												"Error in recording stop event for timing CUDA PST construction code");
							}
							else
							{
								construct_stop_CPU = std::clock();
								construct_stop_wall = std::chrono::steady_clock::now();
							}
						}

						if (tree == nullptr)
						{
							throwErr("Error: Could not allocate memory for priority search tree");
							return;
						}

						std::cout << *tree << '\n';

						size_t num_res_elems = 0;
						PointStructTemplate<T, void, num_IDs> *res_pt_arr;

						if constexpr (timed)
						{
							if constexpr (pst_type == GPU)
							{
								// Start CUDA search timer (i.e. place this event in default stream)
								gpuErrorCheck(cudaEventRecord(search_start_CUDA),
												"Error in recording start event for timing CUDA search code");
							}
							else
							{
								search_start_CPU = std::clock();
								search_start_wall = std::chrono::steady_clock::now();
							}
						}

						// Search/report test phase
						if constexpr (pst_type == GPU)
						{
							if (test_type == LEFT_SEARCH)
							{
								tree->twoSidedLeftSearch(num_res_elems, res_pt_arr,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val,
															warps_per_block);
							}
							else if (test_type == RIGHT_SEARCH)
							{
								tree->twoSidedRightSearch(num_res_elems, res_pt_arr,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val,
															warps_per_block);
							}
							else if (test_type == THREE_SEARCH)
							{
								tree->threeSidedSearch(num_res_elems, res_pt_arr,
														num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
														num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound2,
														num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val,
														warps_per_block);
							}
						}
						else
						{
							if (test_type == LEFT_SEARCH)
							{
								tree->twoSidedLeftSearch(num_res_elems, res_pt_arr,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
							}
							else if (test_type == RIGHT_SEARCH)
							{
								tree->twoSidedRightSearch(num_res_elems, res_pt_arr,
															num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
															num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
							}
							else if (test_type == THREE_SEARCH)
							{
								tree->threeSidedSearch(num_res_elems, res_pt_arr,
														num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound1,
														num_ids_wrapper.tree_type_wrapper.pst_tester.dim1_val_bound2,
														num_ids_wrapper.tree_type_wrapper.pst_tester.min_dim2_val);
							}
						}
						// If test_type == CONSTRUCT, do nothing for the search/report phase

						if constexpr (timed)
						{
							if constexpr (pst_type == GPU)
							{
								// End CUDA search timer
								gpuErrorCheck(cudaEventRecord(search_stop_CUDA),
												"Error in recording stop event for timing CUDA search code");

								// Block CPU execution until search stop event has been recorded
								gpuErrorCheck(cudaEventSynchronize(search_stop_CUDA),
												"Error in blocking CPU execution until completion of stop event for timing CUDA search code");

								// Report construction and search timing
								// Type chosen because of type of parameter of cudaEventElapsedTime
								float ms = 0;	// milliseconds

								gpuErrorCheck(cudaEventElapsedTime(&ms, construct_start_CUDA, construct_stop_CUDA),
												"Error in calculating time elapsed for CUDA PST construction code");
								std::cout << "CUDA PST construction time: " << ms << " ms\n";

								if (test_type != CONSTRUCT)
								{
									gpuErrorCheck(cudaEventElapsedTime(&ms, search_start_CUDA, search_stop_CUDA),
												"Error in calculating time elapsed for CUDA search code");
									std::cout << "CUDA PST search time: " << ms << " ms\n";
								}

								gpuErrorCheck(cudaEventDestroy(construct_start_CUDA),
												"Error in destroying start event for timing CUDA PST construction code");
								gpuErrorCheck(cudaEventDestroy(construct_stop_CUDA),
												"Error in destroying stop event for timing CUDA PST construction code");
								gpuErrorCheck(cudaEventDestroy(search_start_CUDA),
												"Error in destroying start event for timing CUDA search code");
								gpuErrorCheck(cudaEventDestroy(search_stop_CUDA),
												"Error in destroying stop event for timing CUDA search code");
							}
							else
							{
								search_stop_CPU = std::clock();
								search_stop_wall = std::chrono::steady_clock::now();

								std::cout << "CPU PST construction time:\n"
										  << "\tCPU clock time used:\t"
										  << 1000.0 * (construct_stop_CPU - construct_start_CPU) / CLOCKS_PER_SEC << " ms\n"
										  << "\tWall clock time passed:\t"
										  << std::chrono::duration<double, std::milli>(construct_stop_wall - construct_start_wall).count()
										  << " ms\n";


								if (test_type != CONSTRUCT)
								{
									std::cout << "CPU PST search time:\n"
											  << "\tCPU clock time used:\t"
											  << 1000.0 * (search_stop_CPU - search_start_CPU) / CLOCKS_PER_SEC << " ms\n"
											  << "\tWall clock time passed:\t"
											  << std::chrono::duration<double, std::milli>(search_stop_wall - search_start_wall).count()
											  << " ms\n";
								}
							}
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
								throwErr("Error: could not allocate PointStructTemplate<T, IDType, num_IDs> array of size "
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

						delete tree;
						delete[] pt_arr;
						delete[] res_pt_arr;
					};
				};
			};
		};
	};
};


#endif
