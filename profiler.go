package gocudart

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda -lnvrtc
#include <cuda.h>
#include <cudaProfiler.h>
*/
import "C"

func StartProfiler() {
	res := C.cuProfilerStart()
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func End() {
	res := C.cuProfilerStop()
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}
