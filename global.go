package gocudart

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda
// #include <cuda.h>
import "C"

func Init() {
	res := C.cuInit(0)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func GetVersion() int {
	var ver C.int
	var res C.CUresult
	res = C.cuDriverGetVersion(&ver)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return int(ver)
}

func CudaErrorString(res C.CUresult) string {
	switch res {
	case C.CUDA_SUCCESS:
		return "CUDA_SUCCESS"
	case C.CUDA_ERROR_DEINITIALIZED:
		return "CUDA_ERROR_DEINITIALIZED"
	case C.CUDA_ERROR_INVALID_CONTEXT:
		return "CUDA_ERROR_INVALID_CONTEXT"
	case C.CUDA_ERROR_INVALID_HANDLE:
		return "CUDA_ERROR_INVALID_HANDLE"
	case C.CUDA_ERROR_INVALID_IMAGE:
		return "CUDA_ERROR_INVALID_IMAGE"
	case C.CUDA_ERROR_INVALID_VALUE:
		return "CUDA_ERROR_INVALID_VALUE"
	case C.CUDA_ERROR_LAUNCH_FAILED:
		return "CUDA_ERROR_LAUNCH_FAILED"
	case C.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
		return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"
	case C.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
		return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"
	case C.CUDA_ERROR_LAUNCH_TIMEOUT:
		return "CUDA_ERROR_LAUNCH_TIMEOUT"
	case C.CUDA_ERROR_NOT_FOUND:
		return "CUDA_ERROR_NOT_FOUND"
	case C.CUDA_ERROR_NOT_INITIALIZED:
		return "CUDA_ERROR_NOT_INITIALIZED"
	case C.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
		return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"
	default:
		return "Unknown error string"
	}
}
