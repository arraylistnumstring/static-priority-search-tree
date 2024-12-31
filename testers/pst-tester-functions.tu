#include "rand-data-pt-generator.h"

#ifdef DEBUG
#include "print-array.h"
#endif


template <typename PointStruct, typename T, typename StaticPST,
			 PSTType pst_type, bool timed, typename RetType, typename GridDimType
		 >
void datasetTest(const std::string input_file, const unsigned tree_ops_warps_per_block,
					GridDimType pt_grid_dims[Dims::NUM_DIMS], GridDimType metacell_dims[Dims::NUM_DIMS],
					cudaDeviceProp &dev_props, const int num_devs, const int dev_ind)
{
#ifdef DEBUG
	std::cout << "Input file: " << input_file << '\n';
#endif

	// Variables must be outside of conditionals to be accessible in later conditionals
	// For CPU PSTs, GPU -> CPU transfer time of metacells must be taken into consideration for the "construct" cost
	std::clock_t construct_start_CPU, construct_stop_CPU, search_start_CPU, search_stop_CPU;
	std::chrono::time_point<std::chrono::steady_clock> construct_start_wall, construct_stop_wall, search_start_wall, search_stop_wall;

	cudaEvent_t construct_start_CUDA, construct_stop_CUDA, search_start_CUDA, search_stop_CUDA;

	if constexpr (timed && pst_type == GPU)
	{
		gpuErrorCheck(cudaEventCreate(&construct_start_CUDA),
						"Error in creating start event for timing CUDA PST construction code");
		gpuErrorCheck(cudaEventCreate(&construct_stop_CUDA),
						"Error in creating stop event for timing CUDA PST construction code");
		gpuErrorCheck(cudaEventCreate(&search_start_CUDA),
						"Error in creating start event for timing CUDA search code");
		gpuErrorCheck(cudaEventCreate(&search_stop_CUDA),
						"Error in creating stop event for timing CUDA search code");
	}

	GridDimType num_verts = 1;
	for (int i = 0; i < Dims::NUM_DIMS; i++)
		num_verts *= pt_grid_dims[i];

	// Read in vertex array from binary file
	T *vertex_arr = readInVertices<T>(input_file, num_verts);

#ifdef DEBUG
	// Instantiate as separate variable, as attempting a direct substitution of an array initialiser doesn't compile, even if statically cast to an appropriate type
	IDType start_inds[Dims::NUM_DIMS] = {0, 0, 0};
	print3DArray(std::cout, vertex_arr, start_inds, pt_grid_dims);
#endif

	T *vertex_arr_d;

	// Copy vertex_arr to GPU so it's ready for metacell formation and marching cubes
	gpuErrorCheck(cudaMalloc(&vertex_arr_d, num_verts * sizeof(T)),
						"Error in allocating vertex storage array on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ " total devices: "
					);
	// Implicitly synchronous from the host's point of view as only pinned memory can have truly asynchronous cudaMemcpy() calls
	gpuErrorCheck(cudaMemcpy(vertex_arr_d, vertex_arr, num_verts * sizeof(T), cudaMemcpyDefault),
						"Error in copying array of vertices to device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ " total devices: "
					);

	// Prior cudaMemcpy() is staged, if not already written through, so can free vertex_arr
	delete[] vertex_arr;

	size_t num_metacells;

	PointStruct *pt_arr = formMetacellTags<PointStruct>(vertex_arr_d, pt_grid_dims,
															metacell_dims, num_metacells,
															dev_ind, num_devs, dev_props.warpSize
														);

	// Check that GPU memory is sufficiently big for the necessary calculations; num_metacells is now known because of formMetacellTags()
}

template <typename PointStruct, typename T, typename IDType, typename StaticPST,
			PSTType pst_type, bool timed, typename RetType, typename IDDistribInstan,
			typename PSTTester
		>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStruct>
		>::value
void randDataTest(const size_t num_elems, const unsigned warps_per_block,
					PSTTestCodes test_type, PSTTester &pst_tester,
					cudaDeviceProp &dev_props, const int num_devs, const int dev_ind,
					IDDistribInstan *const id_distr_ptr)
{
	PointStruct *pt_arr;

	// Because of instantiation failure when the distribution template template parameter contains void as a type parameter, avoid invoking id_distr_ptr if IDType is void
	if constexpr (std::is_void<IDType>::value)
		pt_arr = generateRandPts<PointStruct, T, void>(num_elems, pst_tester.distr,
														pst_tester.rand_num_eng,
														pst_tester.vals_inc_ordered,
														pst_tester.inter_size_distr_active ? &(pst_tester.inter_size_distr) : nullptr
													);
		else
		pt_arr = generateRandPts<PointStruct, T>(num_elems, pst_tester.distr,
													pst_tester.rand_num_eng,
													pst_tester.vals_inc_ordered,
													pst_tester.inter_size_distr_active ? &(pst_tester.inter_size_distr) : nullptr,
													id_distr_ptr
												);

#ifdef DEBUG
	printArray(std::cout, pt_arr, 0, num_elems);
	std::cout << '\n';
#endif

	// Check that GPU memory is sufficiently big for the necessary calculations
	if constexpr (pst_type == GPU)
	{
		const size_t global_mem_needed = StaticPST::calcGlobalMemNeeded(num_elems);

		if (global_mem_needed > dev_props.totalGlobalMem)
		{
			throwErr("Error: needed global memory space of "
						+ std::to_string(global_mem_needed)
						+ " B required for data structure and processing exceeds limit of global memory = "
						+ std::to_string(dev_props.totalGlobalMem) + " B on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ " total devices: ");
		}
		
		// Set GPU pending kernel queue size limit; note that the queue takes up global memory, hence why a kernel launch that exceeds the queue's capacity may cause an "Invalid __global__ write of n bytes" error message in compute-sanitizer that points to the line of one of its function parameters
		if (num_elems/2 > 2048)		// Default value: 2048
		{
			gpuErrorCheck(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, num_elems/2),
							"Error in increasing pending kernel queue size to "
							+ std::to_string(num_elems/2) + " on device "
							+ std::to_string(dev_ind) + " of "
							+ std::to_string(num_devs) + " total devices: ");
		}
	}

	// Variables must be outside of conditionals to be accessible in later conditionals
	cudaEvent_t construct_start_CUDA, construct_stop_CUDA, search_start_CUDA, search_stop_CUDA;
	std::clock_t construct_start_CPU, construct_stop_CPU, search_start_CPU, search_stop_CPU;
	std::chrono::time_point<std::chrono::steady_clock> construct_start_wall, construct_stop_wall,
														search_start_wall, search_stop_wall;

	if constexpr (timed)
	{
		if constexpr (pst_type == GPU)
		{
			gpuErrorCheck(cudaEventCreate(&construct_start_CUDA),
							"Error in creating start event for timing CUDA PST construction code");
			gpuErrorCheck(cudaEventCreate(&construct_stop_CUDA),
							"Error in creating stop event for timing CUDA PST construction code");
			// For accuracy of measurement of search speeds, create search events here, even if this is a construction-only test
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

	StaticPST *tree;
	if constexpr (pst_type == GPU)
		tree = new StaticPST(pt_arr, num_elems, warps_per_block, dev_ind, num_devs, dev_props);
	else
		tree = new StaticPST(pt_arr, num_elems);

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
			tree->twoSidedLeftSearch(num_res_elems, res_arr, pst_tester.dim1_val_bound1,
										pst_tester.min_dim2_val, warps_per_block);
		}
		else if (test_type == RIGHT_SEARCH)
		{
			tree->twoSidedRightSearch(num_res_elems, res_arr, pst_tester.dim1_val_bound1,
										pst_tester.min_dim2_val, warps_per_block);
		}
		else if (test_type == THREE_SEARCH)
		{
			tree->threeSidedSearch(num_res_elems, res_arr, pst_tester.dim1_val_bound1,
									pst_tester.dim1_val_bound2, pst_tester.min_dim2_val,
									warps_per_block);
		}
	}
	else
	{
		if (test_type == LEFT_SEARCH)
		{
			tree->twoSidedLeftSearch(num_res_elems, res_arr, pst_tester.dim1_val_bound1,
										pst_tester.min_dim2_val);
		}
		else if (test_type == RIGHT_SEARCH)
		{
			tree->twoSidedRightSearch(num_res_elems, res_arr, pst_tester.dim1_val_bound1,
										pst_tester.min_dim2_val);
		}
		else if (test_type == THREE_SEARCH)
		{
			tree->threeSidedSearch(num_res_elems, res_arr, pst_tester.dim1_val_bound1,
									pst_tester.dim1_val_bound2, pst_tester.min_dim2_val);
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
			throwErr("Error: could not allocate PointStruct array of size "
						+ std::to_string(num_res_elems) + " on host");

		// Copy data from res_arr_d to res_arr
		gpuErrorCheck(cudaMemcpy(res_arr, res_arr_d, num_res_elems * sizeof(PointStruct),
									cudaMemcpyDefault), 
						"Error in copying array of PointStruct objects from device "
						+ std::to_string(ptr_info.device) + ": ");

		// Free on-device array of PointStructTemplates
		gpuErrorCheck(cudaFree(res_arr_d), "Error in freeing on-device array of result PointStructs on device "
						+ std::to_string(ptr_info.device) + ": ");
	}

	// Sort output for consistency (specifically for GPU-reported outputs, which may be randomly ordered and therefore must be sorted for easy comparison of results)
	if constexpr (std::is_same<RetType, IDType>::value)
		std::sort(res_arr, res_arr + num_res_elems,
				[](const IDType &id_1, const IDType &id_2)
				{
					return id_1 < id_2;
				});
	else
		std::sort(res_arr, res_arr + num_res_elems,
				[](const PointStruct &pt_1, const PointStruct &pt_2)
				{
					return pt_1.compareDim1(pt_2) < 0;
				});

	printArray(std::cout, res_arr, 0, num_res_elems);
	std::cout << '\n';

	delete tree;
	delete[] res_arr;
	delete[] pt_arr;
}
