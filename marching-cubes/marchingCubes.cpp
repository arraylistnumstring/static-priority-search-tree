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

	// saveToFile();

	cleanup();
}

#define DEBUG_BUFFERS 0

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void computeIsosurface() {
	int threads = 128;
	dim3 grid(numVoxels / threads, 1, 1);

	// get around maximum grid size of 65535 in each dimension
	if (grid.x > 65535) {
		grid.y = grid.x / 32768;
		grid.x = 32768;
	}

	// calculate number of vertices need per voxel
	launch_classifyVoxel(grid, threads, d_voxelVerts, d_voxelOccupied, d_volume,
			gridSize, gridSizeShift, gridSizeMask, numVoxels,
			voxelSize, isoValue);
#if DEBUG_BUFFERS
	printf("voxelVerts:\n");
	dumpBuffer(d_voxelVerts, numVoxels, sizeof(uint));
#endif

#if SKIP_EMPTY_VOXELS
	// scan voxel occupied array
	ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

#if DEBUG_BUFFERS
	printf("voxelOccupiedScan:\n");
	dumpBuffer(d_voxelOccupiedScan, numVoxels, sizeof(uint));
#endif

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
					(void *)(d_voxelOccupied + numVoxels - 1),
					sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
					(void *)(d_voxelOccupiedScan + numVoxels - 1),
					sizeof(uint), cudaMemcpyDeviceToHost));
		activeVoxels = lastElement + lastScanElement;
	}

	if (activeVoxels == 0) {
		// return if there are no full voxels
		totalVerts = 0;
		return;
	}

	// compact voxel index array
	launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied,
			d_voxelOccupiedScan, numVoxels);
	getLastCudaError("compactVoxels failed");

#endif  // SKIP_EMPTY_VOXELS

	// scan voxel vertex count array
	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

#if DEBUG_BUFFERS
	printf("voxelVertsScan:\n");
	dumpBuffer(d_voxelVertsScan, numVoxels, sizeof(uint));
#endif

	// readback total number of vertices
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
					(void *)(d_voxelVerts + numVoxels - 1),
					sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
					(void *)(d_voxelVertsScan + numVoxels - 1),
					sizeof(uint), cudaMemcpyDeviceToHost));
		totalVerts = lastElement + lastScanElement;
	}

	// generate triangles, writing to vertex buffers
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_posvbo_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
				(void **)&d_pos, &num_bytes, cuda_posvbo_resource));
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_normalvbo_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
				(void **)&d_normal, &num_bytes, cuda_normalvbo_resource));

#if SKIP_EMPTY_VOXELS
	dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
#else
	dim3 grid2((int)ceil(numVoxels / (float)NTHREADS), 1, 1);
#endif

	while (grid2.x > 65535) {
		grid2.x /= 2;
		grid2.y *= 2;
	}

#if SAMPLE_VOLUME
	launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
			d_voxelVertsScan, d_volume, gridSize, gridSizeShift,
			gridSizeMask, voxelSize, isoValue, activeVoxels,
			maxVerts);
#else
	launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal, d_compVoxelArray,
			d_voxelVertsScan, gridSize, gridSizeShift,
			gridSizeMask, voxelSize, isoValue, activeVoxels,
			maxVerts);
#endif

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_normalvbo_resource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_posvbo_resource, 0));
}

// shader for displaying floating-point texture
static const char *shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code) {
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
			(GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1) {
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos,
				error_string);
		return 0;
	}

	return program_id;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize OpenGL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv) {
	// Create GL context; may need to replace this portion with EGL or some headless OpenGL context creator
	if (!isGLVersionSupported(2, 0)) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	glEnable(GL_DEPTH_TEST);

	// good old-fashioned fixed function lighting
	float black[] = {0.0f, 0.0f, 0.0f, 1.0f};
	float white[] = {1.0f, 1.0f, 1.0f, 1.0f};
	float ambient[] = {0.1f, 0.1f, 0.1f, 1.0f};
	float diffuse[] = {0.9f, 0.9f, 0.9f, 1.0f};
	float lightPos[] = {0.0f, 0.0f, 1.0f, 0.0f};

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

	glLightfv(GL_LIGHT0, GL_AMBIENT, white);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
	glLightfv(GL_LIGHT0, GL_SPECULAR, white);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);

	// load shader program
	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, unsigned int size) {
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutReportErrors();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource **cuda_resource) {
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	// DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(*vbo));
	cudaGraphicsUnregisterResource(*cuda_resource);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Render isosurface geometry from the vertex buffers
////////////////////////////////////////////////////////////////////////////////
void renderIsosurface() {
	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, normalVbo);
	glNormalPointer(GL_FLOAT, sizeof(float) * 4, 0);
	glEnableClientState(GL_NORMAL_ARRAY);

	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_TRIANGLES, 0, totalVerts);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display() {
	sdkStartTimer(&timer);

	// run CUDA kernel to generate geometry
	if (compute) {
		computeIsosurface();
	}

	// Common display code path
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// set view matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(translate.x, translate.y, translate.z);
		glRotatef(rotate.x, 1.0, 0.0, 0.0);
		glRotatef(rotate.y, 0.0, 1.0, 0.0);

		glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

		if (lighting) {
			glEnable(GL_LIGHTING);
		}

		// render
		if (render) {
			glPushMatrix();
			glRotatef(180.0, 0.0, 1.0, 0.0);
			glRotatef(90.0, 1.0, 0.0, 0.0);
			renderIsosurface();
			glPopMatrix();
		}

		glDisable(GL_LIGHTING);
	}
}
