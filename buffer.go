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
		panic(CudaErrorString(res))
	}
	buf.num = size
	return &buf
}

func (buf *Buffer) MemSet8(v uint8, num int) {
	if num < 0 {
		num = buf.num
	}
	res := C.cuMemsetD8(buf.Id, C.uchar(v), C.size_t(num))
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (buf *Buffer) MemSet16(v uint16, num int) {
	if num < 0 {
		num = buf.num / 2
	}
	res := C.cuMemsetD16(buf.Id, C.ushort(v), C.size_t(num))
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (buf *Buffer) MemSet32(v uint32, num int) {
	if num < 0 {
		num = buf.num / 4
	}
	res := C.cuMemsetD32(buf.Id, C.uint(v), C.size_t(num))
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (buf *Buffer) FromHostN(source unsafe.Pointer, size int) {
	if size > buf.num {
		panic("Trying to copy from host more bytes than buffer capacity")
	}
	res := C.cuMemcpyHtoD(buf.Id, source, C.size_t(size))
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (buf *Buffer) FromHost(source unsafe.Pointer) {
	buf.FromHostN(source, buf.num)
}

func (buf *Buffer) FromDeviceN(dest unsafe.Pointer, size int) {
	if size > buf.num {
		panic("Trying to copy from device more bytes than buffer capacity")
	}
	res := C.cuMemcpyDtoH(dest, buf.Id, C.size_t(size))
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (buf *Buffer) FromDevice(dest unsafe.Pointer) {
	buf.FromDeviceN(dest, buf.num)
}

func AllocManaged(size int) unsafe.Pointer {
	var buf uintptr
	res := C.cuMemAllocManaged((*C.CUdeviceptr)(unsafe.Pointer(&buf)), C.size_t(size), C.CU_MEM_ATTACH_GLOBAL)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return unsafe.Pointer(buf)
}

// Convenience functions to read/copy single variables
func (buf *Buffer) FromIntC(v C.int) {
	buf.FromHostN(unsafe.Pointer(&v), C.sizeof_int)
}

func (buf *Buffer) FromFloatC(v C.float) {
	buf.FromHostN(unsafe.Pointer(&v), C.sizeof_float)
}

func (buf *Buffer) FromDoubleC(v C.double) {
	buf.FromHostN(unsafe.Pointer(&v), C.sizeof_double)
}

func (buf *Buffer) FromInt32(v int32) {
	buf.FromHostN(unsafe.Pointer(&v), 4)
}

func (buf *Buffer) FromInt64(v int64) {
	buf.FromHostN(unsafe.Pointer(&v), 8)
}

func (buf *Buffer) FromFloat32(v float32) {
	buf.FromHostN(unsafe.Pointer(&v), 4)
}

func (buf *Buffer) FromFloat64(v float64) {
	buf.FromHostN(unsafe.Pointer(&v), 8)
}

func (buf *Buffer) ToInt32() int32 {
	var v int32
	buf.FromDeviceN(unsafe.Pointer(&v), 4)
	return v
}

func (buf *Buffer) ToInt64() int64 {
	var v int64
	buf.FromDeviceN(unsafe.Pointer(&v), 8)
	return v
}

func (buf *Buffer) ToFloat32() float32 {
	var v float32
	buf.FromDeviceN(unsafe.Pointer(&v), 4)
	return v
}

func (buf *Buffer) ToFloat64() float64 {
	var v float64
	buf.FromDeviceN(unsafe.Pointer(&v), 8)
	return v
}
