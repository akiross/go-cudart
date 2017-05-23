#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <nvrtc.h>

#define NUM 100000
#define xstr(X) str(X)
#define str(X) #X

const char *program = "// Vector addition\n"
"extern \"C\"\n"
"__global__ void vecSum(int *a, int *b, int *c) {\n"
"	int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"	if (i < " xstr(NUM) ")\n"
"		c[i] = a[i] + b[i];\n"
"}\n";

void errorCheck(CUresult res, char *message) {
#define CASE(X) case X: printf(" Got error: %s\n", #X); break
	if (res != CUDA_SUCCESS) {
		switch (res) {
		CASE(CUDA_ERROR_DEINITIALIZED);
		CASE(CUDA_ERROR_INVALID_CONTEXT);
		CASE(CUDA_ERROR_INVALID_HANDLE);
		CASE(CUDA_ERROR_INVALID_IMAGE);
		CASE(CUDA_ERROR_INVALID_VALUE);
		CASE(CUDA_ERROR_LAUNCH_FAILED);
		CASE(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
		CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
		CASE(CUDA_ERROR_LAUNCH_TIMEOUT);
		CASE(CUDA_ERROR_NOT_FOUND);
		CASE(CUDA_ERROR_NOT_INITIALIZED);
		CASE(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED);
		default: printf("Another error...\n");
		}
		printf(message, res);
		exit(1);
	}
#undef CASE
}

int main(void) {
	errorCheck(cuInit(0), "Error while initializing: %d\n");

	int numDevices;
	errorCheck(cuDeviceGetCount(&numDevices), "Error while getting devices count: %d\n");
	if (numDevices == 0) {
		printf("Error: no CUDA devices available!");
		exit(0);
	}

	CUdevice dev;
	errorCheck(cuDeviceGet(&dev, 0), "Error while getting device 0: %d\n");

	// Write device name
	char name[100];
	errorCheck(cuDeviceGetName(name, 100, dev), "Error while retrieving device name: %d\n");
	printf("Using device %s\n", name);

	int mbx, mby, mbz, mgx, mgy, mgz;
	cuDeviceGetAttribute(&mbx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
	cuDeviceGetAttribute(&mby, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
	cuDeviceGetAttribute(&mbz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
	cuDeviceGetAttribute(&mgx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
	cuDeviceGetAttribute(&mgy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
	cuDeviceGetAttribute(&mgz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
	printf("Max block dim: %d %d %d\n", mbx, mby, mbz);
	printf("Max grid dim: %d %d %d\n", mgx, mgy, mgz);

	int tpb;
	cuDeviceGetAttribute(&tpb, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
	printf("Max threads per block: %d\n", tpb);

	CUcontext ctx;
	errorCheck(cuCtxCreate(&ctx, 0, dev), "Error while creating context: %d\n");

	nvrtcProgram pr1;
	nvrtcResult res = nvrtcCreateProgram(&pr1, program, "vecSum", 0, 0, 0);
	if (res != NVRTC_SUCCESS) {
		printf("Error while creating program: %d\n", res);
		exit(1);
	}

	// Compilation
	res = nvrtcCompileProgram(pr1, 0, 0);
	if (res != NVRTC_SUCCESS) {
		size_t logLen;
		nvrtcGetProgramLogSize(pr1, &logLen);
		char *buf = (char*)malloc(logLen);
		nvrtcGetProgramLog(pr1, buf);
		printf(buf);
		free((void*)buf);
	} else {
		printf("Compilation successful\n");
	}

	// Retrieve PTX
	size_t ptxLen;
	nvrtcGetPTXSize(pr1, &ptxLen);
	char *ptx = (char*)malloc(sizeof(char) * ptxLen);
	nvrtcGetPTX(pr1, ptx);
	nvrtcDestroyProgram(&pr1);

	// Create module
	CUmodule mod;
	errorCheck(cuModuleLoadData(&mod, (void*)ptx), "Error while loading data: %d\n");

	CUfunction func;
	errorCheck(cuModuleGetFunction(&func, mod, "vecSum"), "Error when retrieving func: %d\n");

	// Create stream
	CUstream stream = 0; // Default stream
	//errorCheck(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "Error while creating stream: %d\n");

	// Setup host memory
	int ha[NUM], hb[NUM], hc[NUM];
	size_t size = NUM * sizeof(int);
	for (int i = 0; i < NUM; i++) {
		ha[i] = i+1;
		hb[i] = 10000-i*i;
		hc[i] = -1;
	}

	// Setup device memory
	CUdeviceptr da, db, dc;
	errorCheck(cuMemAlloc(&da, size), "Error while allocating da: %d\n");
	errorCheck(cuMemAlloc(&db, size), "Error while allocating db: %d\n");
	errorCheck(cuMemAlloc(&dc, size), "Error while allocating dc: %d\n");

	errorCheck(cuCtxSynchronize(), "Error while sync: %d\n");

	// Copy from host to device
	errorCheck(cuMemcpyHtoD(da, ha, size), "Error while copying ha->da: %d\n");
	errorCheck(cuMemcpyHtoD(db, hb, size), "Error while copying hb->db: %d\n");

	errorCheck(cuCtxSynchronize(), "Error while sync: %d\n");

	void *args[] = {(void*)&da, (void*)&db, (void*)&dc};
	int bpg = (NUM + tpb - 1) / tpb; // Blocks per grid
	errorCheck(cuLaunchKernel(func,
				/* grid xyz */ bpg, 1, 1,
				/* block xyz */ tpb, 1, 1,
				/* shmem */ 0,
				/* stream */ stream,
				&args[0], 0), "Error while launching kernel: %d\n");
	errorCheck(cuCtxSynchronize(), "Error while sync: %d\n");

	errorCheck(cuMemcpyDtoH(hc, dc, size), "Error while copying back: %d\n");

	cuMemFree(da);
	cuMemFree(db);
	cuMemFree(dc);

	printf("Computation done!\n");
	for (int i = 0; i < 100; i++) {
		int res = ha[i] + hb[i];
		if (hc[i] != res) {
			printf("Test failed at %d: %d + %d = %d != %d\n", i, ha[i], hb[i], res, hc[i]);
			exit(1);
		}
	}
	printf("All test passed :)\n");
}

