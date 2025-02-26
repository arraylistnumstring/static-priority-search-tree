#ifndef GPU_ERR_CHK_H
#define GPU_ERR_CHK_H

#include <string>

#include "err-chk.h"

void gpuErrorCheck(cudaError_t cuda_err, std::string err_str)
{
	if (cuda_err != cudaSuccess)
		throwErr(err_str + cudaGetErrorString(cuda_err));
};

#endif
