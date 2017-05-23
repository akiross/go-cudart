package gocudart

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda -lnvrtc
#include <cuda.h>
*/
import "C"
import "unsafe"

// TODO better API required:
// take advantage of type safety instead of relying on user remebering it
type Buffer struct {
	dev C.CUdeviceptr
	num int
}

func NewInt32Buffer(count int) *Buffer {
	var buf Buffer
	res := C.cuMemAlloc(&buf.dev, 4*C.size_t(count))
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
	buf.num = count
	return &buf
}

func NewFloatBuffer(count int) *Buffer {
	var buf Buffer
	res := C.cuMemAlloc(&buf.dev, C.sizeof_float*C.size_t(count))
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
	buf.num = count
	return &buf
}

func NewDoubleBuffer(count int) *Buffer {
	var buf Buffer
	res := C.cuMemAlloc(&buf.dev, C.sizeof_double*C.size_t(count))
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
	buf.num = count
	return &buf
}

func (buf *Buffer) FloatToDevice(dat []float32) {
	C.cuMemcpyHtoD(buf.dev, unsafe.Pointer(&dat[0]), C.sizeof_float*C.size_t(buf.num))
}

func (buf *Buffer) FloatFromDevice() []float32 {
	dat := make([]float32, buf.num)
	C.cuMemcpyDtoH(unsafe.Pointer(&dat[0]), buf.dev, C.sizeof_float*C.size_t(buf.num))
	return dat
}

func (buf *Buffer) Int32ToDevice(dat []int32) {
	C.cuMemcpyHtoD(buf.dev, unsafe.Pointer(&dat[0]), 4*C.size_t(buf.num))
}

func (buf *Buffer) Int32FromDevice() []int32 {
	dat := make([]int32, buf.num)
	C.cuMemcpyDtoH(unsafe.Pointer(&dat[0]), buf.dev, 4*C.size_t(buf.num))
	return dat
}
