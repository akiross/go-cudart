package gocudart

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda -lnvrtc
#include <cuda.h>
*/
import "C"
import "unsafe"

type Buffer struct {
	Id  C.CUdeviceptr
	num int
}

func NewBuffer(size int) *Buffer {
	var buf Buffer
	res := C.cuMemAlloc(&buf.Id, C.size_t(size))
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
	buf.num = size
	return &buf
}

func (buf *Buffer) FromHost(source unsafe.Pointer) {
	res := C.cuMemcpyHtoD(buf.Id, source, C.size_t(buf.num))
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
}

func (buf *Buffer) FromDevice(dest unsafe.Pointer) {
	res := C.cuMemcpyDtoH(dest, buf.Id, C.size_t(buf.num))
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
}
