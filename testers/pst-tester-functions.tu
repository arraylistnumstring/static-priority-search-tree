#include "rand-data-pt-generator.h"

#ifdef DEBUG
#include "print-array.h"
#endif


//void datasetTest()

template <typename PointStruct, typename T, typename IDType, typename StaticPST,
			PSTType pst_type, bool timed, typename Distrib, typename RandNumEng,
			typename IDDistrib
		>
void randDataTest(const size_t num_elems, const unsigned warps_per_block,
					Distrib &distr, RandNumEng &rand_num_eng,
					bool vals_inc_ordered, Distrib *const inter_size_distr_ptr,
					IDDistrib *const id_distr_ptr, cudaDeviceProp &dev_props,
					const int num_devs, const int dev_ind)
{
	PointStruct *pt_arr = generateRandPts<PointStruct, T, IDType>(num_elems, distr, rand_num_eng,
																	vals_inc_ordered,
																	inter_size_distr_ptr,
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
	PointStruct *res_pt_arr;

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
}
