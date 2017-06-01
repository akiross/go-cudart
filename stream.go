package gocudart

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda -lnvrtc
// #include <cuda.h>
// #include <nvrtc.h>
import "C"

type Stream struct {
	Id C.CUstream
}

func NewStream(flags uint) *Stream {
	var str Stream
	res := C.cuStreamCreate(&str.Id, C.uint(flags))
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return &str
}

func (str *Stream) Destroy() {
	res := C.cuStreamDestroy(str.Id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (str *Stream) Symchronize() {
	res := C.cuStreamSynchronize(str.Id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (str *Stream) Query() bool {
	res := C.cuStreamQuery(str.Id)
	if res == C.CUDA_SUCCESS {
		return true
	} else if res == C.CUDA_ERROR_NOT_READY {
		return false
	} else {
		panic(CudaErrorString(res))
	}
}
