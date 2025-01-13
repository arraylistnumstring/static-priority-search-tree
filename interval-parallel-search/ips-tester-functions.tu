#include <algorithm>	// To use sort()
#include <iostream>

#include "interval-parallel-search.h"
#include "isosurface-data-processing.h"
#include "print-array.h"
#include "rand-data-pt-generator.h"


template <typename PointStruct, typename T, bool timed,
		 	typename RetType, typename GridDimType, typename IPSTester
		 >
void datasetTest(const std::string input_file, const unsigned num_thread_blocks,
					const unsigned threads_per_block, IPSTester &ips_tester,
					GridDimType pt_grid_dims[Dims::NUM_DIMS],
					GridDimType metacell_dims[Dims::NUM_DIMS],
					cudaDeviceProp &dev_props, const int num_devs, const int dev_ind)
{
#ifdef DEBUG
	std::cout << "Input file: " << input_file << '\n';
#endif

	GridDimType num_verts = 1;
	for (int i = 0; i < Dims::NUM_DIMS; i++)
		num_verts *= pt_grid_dims[i];


	GridDimType metacell_grid_dims[Dims::NUM_DIMS];
	size_t num_metacells;

	calcNumMetacells(pt_grid_dims, metacell_dims, metacell_grid_dims, num_metacells);

#ifdef DEBUG
	std::cout << "Number of metacells: " << num_metacells << '\n';
#endif

	// Variables must be outside of conditionals to be accessible in later conditionals
	cudaEvent_t construct_start_CUDA, construct_stop_CUDA, search_start_CUDA, search_stop_CUDA;

	if constexpr (timed)
	{
		gpuErrorCheck(cudaEventCreate(&construct_start_CUDA),
						"Error in creating start event for timing CUDA IPS construction code");
		gpuErrorCheck(cudaEventCreate(&construct_stop_CUDA),
						"Error in creating stop event for timing CUDA IPS construction code");
		gpuErrorCheck(cudaEventCreate(&search_start_CUDA),
						"Error in creating start event for timing CUDA search code");
		gpuErrorCheck(cudaEventCreate(&search_stop_CUDA),
						"Error in creating stop event for timing CUDA search code");
	}

	// Read in vertex array from binary file
	T *vertex_arr = readInVertices<T>(input_file, num_verts);

#ifdef DEBUG_VERTS
	// Instantiate as separate variable, as attempting a direct substitution of an array initialiser doesn't compile, even if statically cast to an appropriate type
	GridDimType start_inds[Dims::NUM_DIMS] = {0, 0, 0};
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


	if constexpr (timed)
		// Start CUDA construction timer (i.e. place this event into default stream)
		gpuErrorCheck(cudaEventRecord(construct_start_CUDA),
						"Error in recording start event for timing CUDA IPS construction code");

	// metacell_tag_arr is on device
	PointStruct *metacell_tag_arr = formMetacellTags<PointStruct>(vertex_arr_d, pt_grid_dims,
																	metacell_dims, metacell_grid_dims,
																	num_metacells, dev_ind, num_devs,
																	dev_props.warpSize
																);

	/*
	PointStruct *metacell_tag_arr = formVoxelTags<PointStruct>(vertex_arr_d, pt_grid_dims,
																metacell_dims, num_metacells,
																dev_ind, num_devs
															);
	*/

	if constexpr (timed)
		// End CUDA construction timer
		gpuErrorCheck(cudaEventRecord(construct_stop_CUDA),
						"Error in recording stop event for timing CUDA IPS construction code: ");


	size_t num_res_elems = 0;
	RetType *res_arr;

	if constexpr (timed)
		// Start CUDA search timer (i.e. place this event in default stream)
		gpuErrorCheck(cudaEventRecord(search_start_CUDA),
						"Error in recording start event for timing CUDA search code: ");

	intervalParallelSearch(metacell_tag_arr, num_metacells, res_arr, num_res_elems,
							ips_tester.search_val, dev_ind, num_devs, dev_props.warpSize,
							num_thread_blocks, threads_per_block);

	if constexpr (timed)
	{
		// End CUDA search timer
		gpuErrorCheck(cudaEventRecord(search_stop_CUDA),
						"Error in recording stop event for timing CUDA search code: ");

		// Block CPU execution until search stop event has been recorded
		gpuErrorCheck(cudaEventSynchronize(search_stop_CUDA),
						"Error in blocking CPU execution until completion of stop event for timing CUDA search code: ");

		// Report construction and search timing
		// Type chosen because of type of parameter of cudaEventElapsedTime
		float ms = 0;	// milliseconds

		gpuErrorCheck(cudaEventElapsedTime(&ms, construct_start_CUDA, construct_stop_CUDA),
						"Error in calculating time elapsed for CUDA IPS construction code: ");
		std::cout << "CUDA IPS construction time: " << ms << " ms\n";

		gpuErrorCheck(cudaEventElapsedTime(&ms, search_start_CUDA, search_stop_CUDA),
						"Error in calculating time elapsed for CUDA search code: ");
		std::cout << "CUDA IPS search time: " << ms << " ms\n";

		gpuErrorCheck(cudaEventDestroy(construct_start_CUDA),
						"Error in destroying start event for timing CUDA IPS construction code: ");
		gpuErrorCheck(cudaEventDestroy(construct_stop_CUDA),
						"Error in destroying stop event for timing CUDA IPS construction code: ");
		gpuErrorCheck(cudaEventDestroy(search_start_CUDA),
						"Error in destroying start event for timing CUDA search code: ");
		gpuErrorCheck(cudaEventDestroy(search_stop_CUDA),
						"Error in destroying stop event for timing CUDA search code: ");
	}

	// If result array is on GPU, copy to CPU and print
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
			throwErr("Error: could not allocate result object array of size "
						+ std::to_string(num_res_elems) + " on host");

		// Copy data from res_arr_d to res_arr
		gpuErrorCheck(cudaMemcpy(res_arr, res_arr_d, num_res_elems * sizeof(RetType),
									cudaMemcpyDefault), 
						"Error in copying array of result objects from device "
						+ std::to_string(ptr_info.device) + ": ");

		// Free on-device array of RetType elements
		gpuErrorCheck(cudaFree(res_arr_d),
						"Error in freeing array of result objects on device "
						+ std::to_string(ptr_info.device) + ": ");

	}

	// Sort output for consistency (specifically compared to GPU-reported outputs, which may be randomly ordered and must therefore be sorted for easy comparisons)
	if constexpr (std::is_same<RetType, GridDimType>::value)
		std::sort(res_arr, res_arr + num_res_elems,
				[](const GridDimType &gd_1, const GridDimType &gd_2)
				{
					return gd_1 < gd_2;
				});
	else
		std::sort(res_arr, res_arr + num_res_elems,
				[](const PointStruct &pt_1, const PointStruct &pt_2)
				{
					return pt_1.compareDim1(pt_2) < 0;
				});

#ifdef DEBUG
	std::cout << "About to report search results\n";
#endif

	printArray(std::cout, res_arr, 0, num_res_elems);
	std::cout << '\n';

#ifdef DEBUG
	std::cout << "Completed reporting of results\n";
#endif

	delete[] res_arr;

	gpuErrorCheck(cudaPointerGetAttributes(&ptr_info, metacell_tag_arr),
					"Error in determining location type of memory address of input metacell tag array (i.e. whether on host or device): ");
	if (ptr_info.type == cudaMemoryTypeDevice)
		gpuErrorCheck(cudaFree(metacell_tag_arr),
						"Error in freeing array of metacell tags on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ " total devices: "
					);
	else
		delete[] metacell_tag_arr;
}


template <typename PointStruct, typename T, typename IDType, bool timed,
		 	typename RetType, typename IDDistribInstan, typename IPSTester
		>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStruct>
		>::value
void randDataTest(const size_t num_elems, const unsigned num_thread_blocks,
					const unsigned threads_per_block, IPSTester &ips_tester,
					cudaDeviceProp &dev_props, const int num_devs, const int dev_ind,
					IDDistribInstan *const id_distr_ptr)
{
	PointStruct *pt_arr;

	// Because of instantiation failure when the distribution template template parameter contains void as a type parameter, avoid invoking id_distr_ptr if IDType is void
	// pt_arr is on host
	if constexpr (std::is_void<IDType>::value)
		pt_arr = generateRandPts<PointStruct, T, void>(num_elems, ips_tester.distr,
														ips_tester.rand_num_eng,
														true,
														ips_tester.inter_size_distr_active ? &(ips_tester.inter_size_distr) : nullptr
													);
	else
		pt_arr = generateRandPts<PointStruct, T>(num_elems, ips_tester.distr,
													ips_tester.rand_num_eng,
													true,
													ips_tester.inter_size_distr_active ? &(ips_tester.inter_size_distr) : nullptr,
													id_distr_ptr
												);

#ifdef DEBUG
	printArray(std::cout, pt_arr, 0, num_elems);
	std::cout << '\n';
#endif

	// Variables must be outside of conditionals to be accessible in later conditionals
	cudaEvent_t construct_start_CUDA, construct_stop_CUDA, search_start_CUDA, search_stop_CUDA;

	if constexpr (timed)
	{
		gpuErrorCheck(cudaEventCreate(&construct_start_CUDA),
						"Error in creating start event for timing CUDA IPS construction code");
		gpuErrorCheck(cudaEventCreate(&construct_stop_CUDA),
						"Error in creating stop event for timing CUDA IPS construction code");
		// For accuracy of measurement of search speeds, create search events here, even if this is a construction-only test
		gpuErrorCheck(cudaEventCreate(&search_start_CUDA),
						"Error in creating start event for timing CUDA search code");
		gpuErrorCheck(cudaEventCreate(&search_stop_CUDA),
						"Error in creating stop event for timing CUDA search code");

		// Start CUDA construction timer (i.e. place this event into default stream)
		// Construction cost of copying data to device must be timed
		gpuErrorCheck(cudaEventRecord(construct_start_CUDA),
						"Error in recording start event for timing CUDA IPS construction code");
	}
	
	// Copy PointStruct array to GPU for IPS to process
	PointStruct *pt_arr_d;
	gpuErrorCheck(cudaMalloc(&pt_arr_d, num_elems * sizeof(PointStruct)),
					"Error in allocating PointStruct storage array on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ " total devices: "
				);
	gpuErrorCheck(cudaMemcpy(pt_arr_d, pt_arr, num_elems * sizeof(PointStruct),
								cudaMemcpyDefault),
					"Error in copying array of PointStructs from host to device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
					+ " total devices: "
				);
	delete[] pt_arr;

	pt_arr = pt_arr_d;

	if constexpr (timed)
		// End CUDA construction timer
		gpuErrorCheck(cudaEventRecord(construct_stop_CUDA),
						"Error in recording stop event for timing CUDA IPS construction code");

	if (num_elems == 0)
	{
		std::cout << "num_elems = 0; nothing to do\n";
		return;
	}

	size_t num_res_elems = 0;
	RetType *res_arr;

	if constexpr (timed)
		// Start CUDA search timer (i.e. place this event in default stream)
		gpuErrorCheck(cudaEventRecord(search_start_CUDA),
						"Error in recording start event for timing CUDA search code");

	// Search/report test phase
	intervalParallelSearch(pt_arr, num_elems, res_arr, num_res_elems,
							ips_tester.search_val, dev_ind, num_devs, dev_props.warpSize,
							num_thread_blocks, threads_per_block);

	if constexpr (timed)
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
						"Error in calculating time elapsed for CUDA IPS construction code");
		std::cout << "CUDA IPS construction time: " << ms << " ms\n";

		gpuErrorCheck(cudaEventElapsedTime(&ms, search_start_CUDA, search_stop_CUDA),
						"Error in calculating time elapsed for CUDA search code");
		std::cout << "CUDA IPS search time: " << ms << " ms\n";

		gpuErrorCheck(cudaEventDestroy(construct_start_CUDA),
						"Error in destroying start event for timing CUDA IPS construction code");
		gpuErrorCheck(cudaEventDestroy(construct_stop_CUDA),
						"Error in destroying stop event for timing CUDA IPS construction code");
		gpuErrorCheck(cudaEventDestroy(search_start_CUDA),
						"Error in destroying start event for timing CUDA search code");
		gpuErrorCheck(cudaEventDestroy(search_stop_CUDA),
						"Error in destroying stop event for timing CUDA search code");
	}

	// If result array is on GPU, copy to CPU and print
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
			throwErr("Error: could not allocate result object array of size "
						+ std::to_string(num_res_elems) + " on host");

		// Copy data from res_arr_d to res_arr
		gpuErrorCheck(cudaMemcpy(res_arr, res_arr_d, num_res_elems * sizeof(RetType),
									cudaMemcpyDefault), 
						"Error in copying array of result objects from device "
						+ std::to_string(ptr_info.device) + ": ");

		// Free on-device array of RetType elements
		gpuErrorCheck(cudaFree(res_arr_d),
						"Error in freeing array of result objects on device "
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

#ifdef DEBUG
	std::cout << "About to report search results\n";
#endif

	printArray(std::cout, res_arr, 0, num_res_elems);
	std::cout << '\n';

#ifdef DEBUG
	std::cout << "Completed reporting of results\n";
#endif

	delete[] res_arr;

	gpuErrorCheck(cudaPointerGetAttributes(&ptr_info, pt_arr),
					"Error in determining location type of memory address of input PointStruct array (i.e. whether on host or device): ");
	if (ptr_info.type == cudaMemoryTypeDevice)
		gpuErrorCheck(cudaFree(pt_arr),
						"Error in freeing array of PointStructs on device "
						+ std::to_string(dev_ind) + " of " + std::to_string(num_devs)
						+ " total devices: "
					);
	else
		delete[] pt_arr;
}
