#ifndef INTERVAL_PARALLEL_SEARCH_H
#define INTERVAL_PARALLEL_SEARCH_H

#include "dev-symbols.h"	// For global memory-scoped variable res_arr_ind_d
#include "gpu-err-chk.h"
#include "warp-shuffles.h"

// Method of Liu et al. (2016): embarrassingly parallel search for active metacells, superficially modified for parity with PST search method

// Given an array of PointStructTemplate<T, IDType, num_IDs>, return an on-device array of either points or IDs, each point pt or ID for point pt satisfies search_val \in [pt.dim1_val, pt.dim2_val])
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
	>::value
void intervalParallelSearch(PointStructTemplate<T, IDType, num_IDs>* pt_arr_d, const size_t num_elems, RetType *&res_arr_d, size_t &num_res_elems, T search_val, const int dev_ind, const int num_devs, const int warp_size, const unsigned num_thread_blocks, const unsigned threads_per_block)
{
	// Allocate space on GPU for output array, whether metacell tags or IDs
	gpuErrorCheck(cudaMalloc(&res_arr_d, num_elems * sizeof(RetType)),
					"Error in allocating array to store PointStruct search result on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	// Set on-device global result array index to 0
	unsigned long long res_arr_ind = 0;
	// Copying to a defined symbol requires use of an extant symbol; note that a symbol is neither a pointer nor a direct data value, but instead the handle by which the variable is denoted, with look-up necessary to generate a pointer if cudaMemcpy() is used (whereas cudaMemcpyToSymbol()/cudaMemcpyFromSymbol() do the lookup and memory copy altogether)
	gpuErrorCheck(cudaMemcpyToSymbol(res_arr_ind_d, &res_arr_ind, sizeof(size_t),
										0, cudaMemcpyDefault),
					"Error in initialising global result array index to 0 on device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");

	const unsigned warps_per_block = threads_per_block / warp_size + ((threads_per_block % warp_size == 0) ? 0 : 1);

#ifdef DEBUG_SEARCH
	std::cout << "Beginning on-device search\n";
#endif

	// Call global function for on-device search
	intervalParallelSearchGlobal<<<num_thread_blocks, threads_per_block, (warps_per_block + 1) * sizeof(unsigned long long)>>>(pt_arr_d, num_elems, res_arr_d, warps_per_block, search_val);
	
	// Because all calls to the device are placed in the same stream (queue) and because cudaMemcpy() is (host-)blocking, this code will not return before the computation has completed
	// res_arr_ind_d points to the next index to write to, meaning that it actually contains the number of elements returned
	gpuErrorCheck(cudaMemcpyFromSymbol(&num_res_elems, res_arr_ind_d,
										sizeof(unsigned long long), 0, cudaMemcpyDefault),
					"Error in copying global result array final index from device "
					+ std::to_string(dev_ind) + " of " + std::to_string(num_devs) + ": ");
};

// Search in parallel for all points pt satisfying search_val \in [pt.dim1_val, pt.dim2_val]; output array is of type IDType if decltype(res_arr_d) = IDType and of type PointStructTemplate<T, IDType, num_IDs> if decltype(res_arr_d) = PointStructTemplate<T, IDType, num_IDs>
// Shared memory must be at least of size (1 + number of warps) * sizeof(unsigned long long), where the 1 stores the block-level offset index of res_arr_d starting at which results are stored (is updated in each iteration)
template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs, typename RetType>
	requires std::disjunction<
						std::is_same<RetType, IDType>,
						std::is_same<RetType, PointStructTemplate<T, IDType, num_IDs>>
	>::value
__global__ void intervalParallelSearchGlobal(PointStructTemplate<T, IDType, num_IDs> *pt_arr_d,
												const size_t num_elems, RetType *const res_arr_d,
												const unsigned warps_per_block, const T search_val)
{
	extern __shared__ unsigned long long s[];
	unsigned long long &block_level_start_ind = *s;
	unsigned long long *warp_level_num_elems_arr = s + 1;

	// Liu et al. kernel; iterate over all PointStructTemplate<T, IDType, num_IDs> elements in pt_arr_d
	// Due to presence of __syncthreads() calls within for loop, whole block must iterate if at least one thread has an element to process
	// Loop unrolling, as number of loops is known explicitly when kernel is called
#pragma unroll
	for (unsigned long long i = blockIdx.x * blockDim.x;
			i < num_elems; i += gridDim.x * blockDim.x)
	{
		unsigned long long thread_level_num_elems = 0;	// Calculated with inclusive prefix sum
		unsigned long long thread_level_offset;		// Calculated with exclusive prefix sum	(i.e. preceding element of inclusive prefix sum result)

		// Needs a separate flag, as thread_level_num_elems will not ultimately be 0 as long as there is at least one preceding thread with an active cell assigned
		bool cell_active = false;

		// Generate mask for threads active during intrawarp phase
		unsigned intrawarp_mask = __ballot_sync(0xffffffff, i + threadIdx.x < num_elems);

		if (i + threadIdx.x < num_elems)	// Intrawarp condition
		{
			// Evaluate if current metacell is active; if active, set corresponding flags and integers to signal to warp-scan (prefix sum)
			if (pt_arr_d[i + threadIdx.x].dim1_val <= search_val
					&& search_val <= pt_arr_d[i + threadIdx.x].dim2_val)
			{
				thread_level_num_elems = 1;
				cell_active = true;
			}

			// Warp-shuffle procedure to calculate inclusive prefix sum, i.e. so current thread knows offset with respect to start index in res_arr_d at which to output result

			// Intrawarp prefix sum
			warpPrefixSum(intrawarp_mask, thread_level_num_elems);

			// Exclusive prefix sum result is simply the element in the preceding index of the result of the inclusive prefix sum; note that as each thread is responsible for at most 1 output element, this is effectively thread_level_num_elems - 1_{cell_active}, where 1_{cell_active} is the indicator function for whether the thread's own value of cell_active is true
			// Use of __shfl_up_sync() in this instance is for generality and for ease of calculation
			thread_level_offset = __shfl_up_sync(intrawarp_mask, thread_level_num_elems, 1);

			// First thread in each warp read from its own value, so reset the offset
			if (threadIdx.x % warpSize == 0)
				thread_level_offset = 0;

			// Last active thread in warp puts total number of slots necessary for all active cells in the warp_level_num_elems_arr shared memory array
			if (threadIdx.x % warpSize == fls(intrawarp_mask))
				// Place total number of elements in this warp at the slot assigned to this warp in shared memory array warp_level_num_elems_arr
				warp_level_num_elems_arr[threadIdx.x / warpSize] = thread_level_num_elems;
		}

		__syncthreads();	// Warp-level info must be ready to use at the block level

		// Interwarp prefix sum (block-level)
		// Being a prefix sum, only one warp should be active in this process for speed and correctness; use the first warp, which is guaranteed to exist

		// Calculate the number of active warps in this iteration of the grid; type of dim3 is a 4-byte triple
		unsigned warps_per_block_curr_iter;

		// Check if all warps in block were active during intrawarp prefix sum
		// As all numbers here are cardinal, there is no off-by-1 error to compensate for here
		if (i + blockDim.x <= num_elems)
			warps_per_block_curr_iter = warps_per_block;
		else
			warps_per_block_curr_iter = (num_elems - i) / warpSize
											+ ( ( (num_elems - i) % warpSize == 0 ) ? 0 : 1);

		// Boundary condition: if only one warp's worth of data needs to be processed in this interation, only one warp is currently active anyway, so no interwarp prefix sum is necessary; however, as the interwarp __shfl_*sync() loop is hence automatically skipped over because of its own enclosing boundary condition, there is no need to explicitly do anything about it here
		if (threadIdx.x / warpSize == 0)
		{
			// If necessary, repeat interwarp prefix sum with base offset of previous iteration until all warps have had their prefix sum calculated
			// Though would be fine to put __ballot_sync() call in loop initialiser here, as there are currently no __syncwarp() calls within the loop that could cause hang, put within loop anyway in case __syncwarp() calls turn out to be necessary
#pragma unroll
			for (unsigned j = 0; j < warps_per_block_curr_iter; j += warpSize)
			{
				// Simultaneously serves as a syncwarp call to ensure that writes and reads are separated, as well as a mask generator for interwarp prefix sum
				unsigned interwarp_mask = __ballot_sync(0xffffffff, threadIdx.x < warps_per_block_curr_iter);

				// Inter-warp condition
				if (threadIdx.x < warps_per_block_curr_iter)
				{
					// If this is not the first set of warps being processed, set the warp group's offset to be the largest result of the previous iteration's interwarp prefix sum
					unsigned long long warp_gp_offset = (j == 0) ? 0 :
																warp_level_num_elems_arr[j - 1];

					unsigned long long warp_level_num_elems = warp_level_num_elems_arr[j + threadIdx.x];

					// Do inclusive prefix sum on warp-level values
					// Interwarp prefix sum
					warpPrefixSum(interwarp_mask, warp_level_num_elems);

					// Store result in shared memory, including the effect of the offset
					warp_level_num_elems_arr[j + threadIdx.x] = warp_gp_offset + warp_level_num_elems;
				}
			}
		}

		// Total number of slots needed to store all active metacells processed by block in current iteration is now known
		__syncthreads();

		// Single thread in block allocates space in res_arr_d with atomic operation
		if (threadIdx.x == 0)
		{
			const unsigned long long block_level_num_elems = warp_level_num_elems_arr[warps_per_block_curr_iter - 1];
			block_level_start_ind = atomicAdd(&res_arr_ind_d, block_level_num_elems);
		}

		unsigned long long warp_level_offset;		// Calculated with exclusive prefix sum

		// All warps acquire their warp-level offset, which is the element of index (warp ID - 1) in warp_level_num_elems_arr
		// Acquisition of warp-level offset is independent of memory allocation, so can occur anytime after the warp-level __syncthreads() call and before writing results to global memory; placed here for optimisation purposes, as all threads other than thread 0 would otherwise be idle between these __syncthreads() calls
		if (threadIdx.x / warpSize == 0)
			warp_level_offset = 0;
		else
			warp_level_offset = warp_level_num_elems_arr[threadIdx.x / warpSize - 1];

		__syncthreads();	// Block-level offset now known

		// Output to result array
		if (cell_active)
		{
			if constexpr (std::is_same<RetType, IDType>::value)
				// Add ID to array
				res_arr_d[block_level_start_ind + warp_level_offset + thread_level_offset]
					= pt_arr_d[i + threadIdx.x].id;
			else
				// Add PtStructTemplate<T, IDType, num_IDs> to array
				res_arr_d[block_level_start_ind + warp_level_offset + thread_level_offset]
					= pt_arr_d[i + threadIdx.x];
		}
	}
};

#endif
