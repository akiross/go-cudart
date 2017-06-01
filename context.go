package gocudart

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda -lnvrtc
// #include <cuda.h>
// #include <nvrtc.h>
import "C"

type Context struct {
	id C.CUcontext
}

// WARNING: The created context is bound to a CPU thread, therefore use runtime.LockOSThread()
func Create(dev *Device, flags uint) *Context {
	var ctx Context
	res := C.cuCtxCreate(&ctx.id, C.uint(flags), dev.id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return &ctx
}

func (ctx *Context) Destroy() {
	res := C.cuCtxDestroy(ctx.id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (ctx *Context) GetApiVersion() uint {
	var ver C.uint
	res := C.cuCtxGetApiVersion(ctx.id, &ver)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return uint(ver)
}

func (ctx *Context) Synchronize() {
	res := C.cuCtxSynchronize()
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func (ctx *Context) PushCurrent() {
	res := C.cuCtxPushCurrent(ctx.id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
}

func PopCurrent() *Context {
	var ctx Context
	res := C.cuCtxPopCurrent(&ctx.id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return &ctx
}
