/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _MARCHING_CUBES_KERNEL_CU_
#define _MARCHING_CUBES_KERNEL_CU_

// C headers
#include <string.h>	// memset()

// Helper code found in Common/ folder of NVIDIA sample code
#include <helper_cuda.h>	// checkCudaErrors()
#include <helper_math.h>	// cross(), lerp(), min()

#include <thrust/device_vector.h>	// thrust::device_ptr type
#include <thrust/scan.h>

// Contains preprocessor variables SKIP_EMPTY_VOXELS and USE_SHARED, as well as typedefs for uint and uchar
#include "defines.h"
#include "tables.h"	// Marching cubes lookup table

// textures containing look-up tables
cudaTextureObject_t triTex;
cudaTextureObject_t numVertsTex;

// volume data
cudaTextureObject_t volumeTex;

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable,
		uint **d_numVertsTable) {
	checkCudaErrors(cudaMalloc((void **)d_edgeTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable,
				256 * sizeof(uint), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

	checkCudaErrors(cudaMalloc((void **)d_triTable, 256 * 16 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable,
				256 * 16 * sizeof(uint), cudaMemcpyHostToDevice));

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = *d_triTable;
	texRes.res.linear.sizeInBytes = 256 * 16 * sizeof(uint);
	texRes.res.linear.desc = channelDesc;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&triTex, &texRes, &texDescr, NULL));

	checkCudaErrors(cudaMalloc((void **)d_numVertsTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable,
				256 * sizeof(uint), cudaMemcpyHostToDevice));

	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = *d_numVertsTable;
	texRes.res.linear.sizeInBytes = 256 * sizeof(uint);
	texRes.res.linear.desc = channelDesc;

	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(
			cudaCreateTextureObject(&numVertsTex, &texRes, &texDescr, NULL));
}

extern "C" void createVolumeTexture(uchar *d_volume, size_t buffSize) {
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = d_volume;
	texRes.res.linear.sizeInBytes = buffSize;
	texRes.res.linear.desc =
		cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(
			cudaCreateTextureObject(&volumeTex, &texRes, &texDescr, NULL));
}

extern "C" void destroyAllTextureObjects() {
	checkCudaErrors(cudaDestroyTextureObject(triTex));
	checkCudaErrors(cudaDestroyTextureObject(numVertsTex));
	checkCudaErrors(cudaDestroyTextureObject(volumeTex));
}

// sample volume data set at a point
__device__ float sampleVolume(cudaTextureObject_t volumeTex, uchar *data,
		uint3 p, uint3 gridSize) {
	p.x = min(p.x, gridSize.x - 1);
	p.y = min(p.y, gridSize.y - 1);
	p.z = min(p.z, gridSize.z - 1);
	uint i = (p.z * gridSize.x * gridSize.y) + (p.y * gridSize.x) + p.x;
	//    return (float) data[i] / 255.0f;
	return tex1Dfetch<float>(volumeTex, i);
}

// compute position in 3d grid from 1d index
// only works for power of 2 sizes
__device__ uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask) {
	uint3 gridPos;
	gridPos.x = i & gridSizeMask.x;
	gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
	gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
	return gridPos;
}

// classify voxel based on number of vertices it will generate
// one thread per voxel
__global__ void classifyVoxel(uint *voxelVerts, uint *voxelOccupied,
		uchar *volume, uint3 gridSize,
		uint3 gridSizeShift, uint3 gridSizeMask,
		uint numVoxels, float3 voxelSize, float isoValue,
		cudaTextureObject_t numVertsTex,
		cudaTextureObject_t volumeTex) {
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

	// read field values at neighbouring grid vertices
	float field[8];
	field[0] = sampleVolume(volumeTex, volume, gridPos, gridSize);
	field[1] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 0, 0), gridSize);
	field[2] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 1, 0), gridSize);
	field[3] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 1, 0), gridSize);
	field[4] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 0, 1), gridSize);
	field[5] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 0, 1), gridSize);
	field[6] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 1, 1), gridSize);
	field[7] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 1, 1), gridSize);

	// calculate flag indicating if each vertex is inside or outside isosurface
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// read number of vertices from texture
	uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

	if (i < numVoxels) {
		voxelVerts[i] = numVerts;
		voxelOccupied[i] = (numVerts > 0);
	}
}

extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts,
		uint *voxelOccupied, uchar *volume,
		uint3 gridSize, uint3 gridSizeShift,
		uint3 gridSizeMask, uint numVoxels,
		float3 voxelSize, float isoValue) {
	// calculate number of vertices need per voxel
	classifyVoxel<<<grid, threads>>>(voxelVerts, voxelOccupied, volume, gridSize,
			gridSizeShift, gridSizeMask, numVoxels,
			voxelSize, isoValue, numVertsTex, volumeTex);
	getLastCudaError("classifyVoxel failed");
}

// compact voxel array
__global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied,
		uint *voxelOccupiedScan, uint numVoxels) {
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (voxelOccupied[i] && (i < numVoxels)) {
		compactedVoxelArray[voxelOccupiedScan[i]] = i;
	}
}

extern "C" void launch_compactVoxels(dim3 grid, dim3 threads,
		uint *compactedVoxelArray,
		uint *voxelOccupied,
		uint *voxelOccupiedScan, uint numVoxels) {
	compactVoxels<<<grid, threads>>>(compactedVoxelArray, voxelOccupied,
			voxelOccupiedScan, numVoxels);
	getLastCudaError("compactVoxels failed");
}

// compute interpolated vertex along an edge; used in generateTriangles2(), i.e. with input volumetric data rather than a field function
__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0,
		float f1) {
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

// calculate triangle normal
__device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2) {
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;
	// note - it's faster to perform normalization in vertex shader rather than
	// here
	return cross(edge0, edge1);
}

// version that calculates flat surface normal for each triangle from input data
__global__ void generateTriangles2(
		float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
		uchar *volume, uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
		float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts,
		cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex,
		cudaTextureObject_t volumeTex) {
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (i > activeVoxels - 1) {
		i = activeVoxels - 1;
	}


#if SKIP_EMPTY_VOXELS
	uint voxel = compactedVoxelArray[i];
#else
	uint voxel = i;
#endif

	// compute position in 3d grid
	uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

	float3 p;
	p.x = -1.0f + (gridPos.x * voxelSize.x);
	p.y = -1.0f + (gridPos.y * voxelSize.y);
	p.z = -1.0f + (gridPos.z * voxelSize.z);

	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

	float field[8];
	field[0] = sampleVolume(volumeTex, volume, gridPos, gridSize);
	field[1] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 0, 0), gridSize);
	field[2] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 1, 0), gridSize);
	field[3] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 1, 0), gridSize);
	field[4] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 0, 1), gridSize);
	field[5] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 0, 1), gridSize);
	field[6] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(1, 1, 1), gridSize);
	field[7] =
		sampleVolume(volumeTex, volume, gridPos + make_uint3(0, 1, 1), gridSize);
	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// find the vertices where the surface intersects the cube

#if USE_SHARED
	// use shared memory to avoid using local
	__shared__ float3 vertlist[12 * NTHREADS];

	vertlist[threadIdx.x] =
		vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[NTHREADS + threadIdx.x] =
		vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[(NTHREADS * 2) + threadIdx.x] =
		vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[(NTHREADS * 3) + threadIdx.x] =
		vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
	vertlist[(NTHREADS * 4) + threadIdx.x] =
		vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[(NTHREADS * 5) + threadIdx.x] =
		vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[(NTHREADS * 6) + threadIdx.x] =
		vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[(NTHREADS * 7) + threadIdx.x] =
		vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
	vertlist[(NTHREADS * 8) + threadIdx.x] =
		vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[(NTHREADS * 9) + threadIdx.x] =
		vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[(NTHREADS * 10) + threadIdx.x] =
		vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[(NTHREADS * 11) + threadIdx.x] =
		vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
	__syncthreads();
#else

	float3 vertlist[12];

	vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

	vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

	vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
#endif

	// output triangle vertices
	uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

	for (int i = 0; i < numVerts; i += 3) {
		uint index = numVertsScanned[voxel] + i;

		float3 *v[3];
		uint edge;
		edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + i);
#if USE_SHARED
		v[0] = &vertlist[(edge * NTHREADS) + threadIdx.x];
#else
		v[0] = &vertlist[edge];
#endif

		edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + i + 1);
#if USE_SHARED
		v[1] = &vertlist[(edge * NTHREADS) + threadIdx.x];
#else
		v[1] = &vertlist[edge];
#endif

		edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + i + 2);
#if USE_SHARED
		v[2] = &vertlist[(edge * NTHREADS) + threadIdx.x];
#else
		v[2] = &vertlist[edge];
#endif

		// calculate triangle surface normal
		float3 n = calcNormal(v[0], v[1], v[2]);

		if (index < (maxVerts - 3)) {
			pos[index] = make_float4(*v[0], 1.0f);
			norm[index] = make_float4(n, 0.0f);

			pos[index + 1] = make_float4(*v[1], 1.0f);
			norm[index + 1] = make_float4(n, 0.0f);

			pos[index + 2] = make_float4(*v[2], 1.0f);
			norm[index + 2] = make_float4(n, 0.0f);
		}
	}
}

extern "C" void launch_generateTriangles2(
		dim3 grid, dim3 threads, float4 *pos, float4 *norm,
		uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize,
		float isoValue, uint activeVoxels, uint maxVerts) {
	generateTriangles2<<<grid, NTHREADS>>>(
			pos, norm, compactedVoxelArray, numVertsScanned, volume, gridSize,
			gridSizeShift, gridSizeMask, voxelSize, isoValue, activeVoxels, maxVerts,
			triTex, numVertsTex, volumeTex);
	getLastCudaError("generateTriangles2 failed");
}

extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
		unsigned int numElements) {
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
			thrust::device_ptr<unsigned int>(input + numElements),
			thrust::device_ptr<unsigned int>(output));
}

#endif
