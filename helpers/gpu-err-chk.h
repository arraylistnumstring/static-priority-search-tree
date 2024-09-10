#ifndef GPU_ERR_CHK_H
#define GPU_ERR_CHK_H

#include <stdexcept>	// To use std::runtime_error

#include "err-chk.h"

void gpuErrorCheck(cudaError_t cuda_err, std::string err_str)
{
	if (cuda_err != cudaSuccess)
		throwErr(err_str + cudaGetErrorString(cuda_err));
};

#endif
