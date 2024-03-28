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

/*
   Marching cubes

   This sample extracts a geometric isosurface from a volume dataset using
   the marching cubes algorithm. It uses the scan (prefix sum) function from
   the Thrust library to perform stream compaction.  Similar techniques can
   be used for other problems that require a variable-sized output per
   thread.

   For more information on marching cubes see:
http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
http://en.wikipedia.org/wiki/Marching_cubes

Volume data courtesy:
http://www9.informatik.uni-erlangen.de/External/vollib/

For more information on the Thrust library
http://code.google.com/p/thrust/

The algorithm consists of several stages:

1. Execute "classifyVoxel" kernel
This evaluates the volume at the corners of each voxel and computes the
number of vertices each voxel will generate.
It is executed using one thread per voxel.
It writes two arrays - voxelOccupied and voxelVertices to global memory.
voxelOccupied is a flag indicating if the voxel is non-empty.

2. Scan "voxelOccupied" array (using Thrust scan)
Read back the total number of occupied voxels from GPU to CPU.
This is the sum of the last value of the exclusive scan and the last
input value.

3. Execute "compactVoxels" kernel
This compacts the voxelOccupied array to get rid of empty voxels.
This allows us to run the complex "generateTriangles" kernel on only
the occupied voxels.

4. Scan voxelVertices array
This gives the start address for the vertex data for each voxel.
We read back the total number of vertices generated from GPU to CPU.

Note that by using a custom scan function we could combine the above two
scan operations above into a single operation.

5. Execute "generateTriangles" kernel
This runs only on the occupied voxels.
It looks up the field values again and generates the triangle data,
using the results of the scan to write the output to the correct addresses.
The marching cubes look-up tables are stored in 1D textures.

6. Render geometry
Using number of vertices from readback.
*/

/*
   Modification by Brian H. Chiang

   Runs marching cubes on single MC cells within parameter metacells and outputs image to a file. This differs from the original, which ran interactively in a GLUT window or calculated arrays and verified with the provided result files. Accordingly, unused or unnecessary definitions have been removed from this iteration of Marching Cubes.
   */

extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts,
		uint *voxelOccupied, uchar *volume,
		uint3 gridSize, uint3 gridSizeShift,
		uint3 gridSizeMask, uint numVoxels,
		float3 voxelSize, float isoValue);

extern "C" void launch_compactVoxels(dim3 grid, dim3 threads,
		uint *compactedVoxelArray,
		uint *voxelOccupied,
		uint *voxelOccupiedScan, uint numVoxels);

extern "C" void launch_generateTriangles2(
		dim3 grid, dim3 threads, float4 *pos, float4 *norm,
		uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
		uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, float3 voxelSize,
		float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable,
		uint **d_numVertsTable);
extern "C" void createVolumeTexture(uchar *d_volume, size_t buffSize);
extern "C" void destroyAllTextureObjects();
extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input,
		unsigned int numElements);

uint3 gridSizeLog2 = make_uint3(5, 5, 5);
uint3 gridSizeShift;
uint3 gridSize;
uint3 gridSizeMask;

float3 voxelSize;
uint numVoxels = 0;
uint maxVerts = 0;
uint activeVoxels = 0;
uint totalVerts = 0;

float isoValue = 0.2f;

// device data
GLuint posVbo, normalVbo;
GLint gl_Shader;
struct cudaGraphicsResource *cuda_posvbo_resource,
							*cuda_normalvbo_resource;  // handles OpenGL-CUDA exchange

float4 *d_pos = 0, *d_normal = 0;

uchar *d_volume = 0;
uint *d_voxelVerts = 0;
uint *d_voxelVertsScan = 0;
uint *d_voxelOccupied = 0;
uint *d_voxelOccupiedScan = 0;
uint *d_compVoxelArray;

// tables
uint *d_numVertsTable = 0;
uint *d_edgeTable = 0;
uint *d_triTable = 0;

// toggles
bool wireframe = false;
// No animation variable, as the goal is to do single-frame image renders
bool lighting = true;
bool render = true;
bool compute = true;

// forward declarations
void runGraphicsTest(int argc, char **argv);
// Removed: runAutoTest(): for output verification only
void initMC(int argc, char **argv);
void computeIsosurface();
// Removed: dumpFile(): for output verification only

// For preprocessor DEBUG-delimited statements
template <class T>
void dumpBuffer(T *d_buffer, int nelements, int size_element);

void cleanup();

bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, unsigned int size);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_resource);

void display();
// Remainder in the original block are GLUT callback functions for interactivity

template <class T>
void dumpBuffer(T *d_buffer, int nelements, int size_element) {
	uint bytes = nelements * size_element;
	T *h_buffer = (T *)malloc(bytes);
	checkCudaErrors(
			cudaMemcpy(h_buffer, d_buffer, bytes, cudaMemcpyDeviceToHost));

	for (int i = 0; i < nelements; i++) {
		printf("%d: %u\n", i, h_buffer[i]);
	}

	printf("\n");
	free(h_buffer);
}

////////////////////////////////////////////////////////////////////////////////
// initialize marching cubes
////////////////////////////////////////////////////////////////////////////////
void initMC(int argc, char **argv) {
	// parse command line arguments
	int n;

	if (checkCmdLineFlag(argc, (const char **)argv, "grid")) {
		n = getCmdLineArgumentInt(argc, (const char **)argv, "grid");
		gridSizeLog2.x = gridSizeLog2.y = gridSizeLog2.z = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "gridx")) {
		n = getCmdLineArgumentInt(argc, (const char **)argv, "gridx");
		gridSizeLog2.x = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "gridy")) {
		n = getCmdLineArgumentInt(argc, (const char **)argv, "gridy");
		gridSizeLog2.y = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "gridz")) {
		n = getCmdLineArgumentInt(argc, (const char **)argv, "gridz");
		gridSizeLog2.z = n;
	}

	char *filename;

	if (getCmdLineArgumentString(argc, (const char **)argv, "file", &filename)) {
		volumeFilename = filename;
	}

	gridSize =
		make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
	gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
	gridSizeShift =
		make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

	numVoxels = gridSize.x * gridSize.y * gridSize.z;
	voxelSize =
		make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
	maxVerts = gridSize.x * gridSize.y * 100;

	printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z,
			numVoxels);
	printf("max verts = %d\n", maxVerts);

#if SAMPLE_VOLUME
	// load volume data
	char *path = sdkFindFilePath(volumeFilename, argv[0]);

	if (path == NULL) {
		fprintf(stderr, "Error finding file '%s'\n", volumeFilename);

		exit(EXIT_FAILURE);
	}

	int size = gridSize.x * gridSize.y * gridSize.z * sizeof(uchar);
	uchar *volume = loadRawFile(path, size);
	checkCudaErrors(cudaMalloc((void **)&d_volume, size));
	checkCudaErrors(cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice));
	free(volume);

	createVolumeTexture(d_volume, size);
#endif

	// create VBOs
	createVBO(&posVbo, maxVerts * sizeof(float) * 4);
	// DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(posVbo) );
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(
				&cuda_posvbo_resource, posVbo, cudaGraphicsMapFlagsWriteDiscard));

	createVBO(&normalVbo, maxVerts * sizeof(float) * 4);
	// DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(normalVbo));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(
				&cuda_normalvbo_resource, normalVbo, cudaGraphicsMapFlagsWriteDiscard));

	// allocate textures
	allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

	// allocate device memory
	unsigned int memSize = sizeof(uint) * numVoxels;
	checkCudaErrors(cudaMalloc((void **)&d_voxelVerts, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelVertsScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupied, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupiedScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_compVoxelArray, memSize));
}

void cleanup()
{
	deleteVBO(&posVbo, &cuda_posvbo_resource);
	deleteVBO(&normalVbo, &cuda_normalvbo_resource);

	destroyAllTextureObjects();
	checkCudaErrors(cudaFree(d_edgeTable));
	checkCudaErrors(cudaFree(d_triTable));
	checkCudaErrors(cudaFree(d_numVertsTable));

	checkCudaErrors(cudaFree(d_voxelVerts));
	checkCudaErrors(cudaFree(d_voxelVertsScan));
	checkCudaErrors(cudaFree(d_voxelOccupied));
	checkCudaErrors(cudaFree(d_voxelOccupiedScan));
	checkCudaErrors(cudaFree(d_compVoxelArray));

	if (d_volume) {
		checkCudaErrors(cudaFree(d_volume));
	}
}

void runGraphicsTest(int argc, char **argv) {
	printf("MarchingCubes\n");

	if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
		printf("[%s]\n", argv[0]);
		printf("   Does not explicitly support -device=n in OpenGL mode\n");
		printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
		printf(" > %s -device=n -file=<reference> -dump=<0/1/2>\n", argv[0]);
		exit(EXIT_SUCCESS);
	}

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA
	// interop.
	// initGL returns false on failure
	if (!initGL(&argc, argv))
		return;


	findCudaDevice(argc, (const char **)argv);


	// Initialize CUDA buffers for Marching Cubes
	initMC(argc, argv);

	display();

	cleanup();
}
