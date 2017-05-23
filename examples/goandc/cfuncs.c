#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <nvrtc.h>

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

void performCompute(void *f, int *ha, int *hb, int *hc, int len) {
	CUfunction func = *(CUfunction*)f;
	// Create stream
	CUstream stream = 0; // Default stream
	//errorCheck(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "Error while creating stream: %d\n");

	// Setup device memory
	CUdeviceptr da, db, dc;
	errorCheck(cuMemAlloc(&da, sizeof(int) * len), "Error while allocating da: %d\n");
	errorCheck(cuMemAlloc(&db, sizeof(int) * len), "Error while allocating db: %d\n");
	errorCheck(cuMemAlloc(&dc, sizeof(int) * len), "Error while allocating dc: %d\n");

	// Copy from host to device
	errorCheck(cuMemcpyHtoD(da, ha, sizeof(int) * len), "Error while copying ha->da: %d\n");
	errorCheck(cuMemcpyHtoD(db, hb, sizeof(int) * len), "Error while copying hb->db: %d\n");

	void *args[] = {&da, &db, &dc, &len};
	int tpb = 256;
	int bpg = (len + tpb - 1) / tpb;
	errorCheck(cuLaunchKernel(func,
				/* grid xyz */ bpg, 1, 1,
				/* block xyz */ tpb, 1, 1,
				/* shmem */ 0,
				/* stream */ stream,
				args, 0), "Error while launching kernel: %d\n");

	errorCheck(cuMemcpyDtoH(hc, dc, sizeof(int) * len), "Error while copying back: %d\n");
	printf("Finished compute of %d elements\n", len);
}

