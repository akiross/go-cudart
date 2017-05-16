package main

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda -lnvrtc
#include <cuda.h>
#include <nvrtc.h>
*/
import "C"
import "unsafe"

type Module struct {
	id C.CUmodule
}

type Function struct {
	id C.CUfunction
}

func CreateModule() *Module {
	return &Module{}
}

func (mod *Module) LoadData(prog *Program) {
	res := C.cuModuleLoadData(&mod.id, unsafe.Pointer(&prog.PTX[0]))
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
}

func (mod *Module) GetFunction(name string) *Function {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	var fun Function
	res := C.cuModuleGetFunction(&fun.id, mod.id, cname)
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
	return &fun
}

type Program struct {
	prog C.nvrtcProgram
	//Source  Source
	//Headers []Source
	PTX []byte
}

type Source struct {
	Source string
	Name   string
}

func GetNVRTCVersion() (major, minor int) {
	var maj, min C.int
	res := C.nvrtcVersion(&maj, &min)
	if res != C.NVRTC_SUCCESS {
		panic(res)
	}
	major, minor = int(maj), int(min)
	return
}

func CreateProgram(src Source, headers []Source) *Program {
	var prog Program

	c_src, c_name := C.CString(src.Source), C.CString(src.Name)
	defer C.free(unsafe.Pointer(c_src))
	defer C.free(unsafe.Pointer(c_name))

	var numHeads C.int = C.int(len(headers))

	var heads, headNames **C.char
	if numHeads == 0 {
		nullptr := (**C.char)(unsafe.Pointer(uintptr(0)))
		heads, headNames = nullptr, nullptr
	} else {
		h_srcs := make([]*C.char, numHeads)
		h_names := make([]*C.char, numHeads)
		for i := 0; i < int(numHeads); i++ {
			h_srcs[i] = C.CString(headers[i].Source)
			defer C.free(unsafe.Pointer(h_srcs[i]))
			h_names[i] = C.CString(headers[i].Name)
			defer C.free(unsafe.Pointer(h_names[i]))
		}
		heads = (**C.char)(unsafe.Pointer(&h_srcs[0]))
		headNames = (**C.char)(unsafe.Pointer(&h_names[0]))
	}
	res := C.nvrtcCreateProgram(&prog.prog, c_src, c_name, numHeads, heads, headNames)
	if res != C.NVRTC_SUCCESS {
		panic(res)
	}

	return &prog
}

func (prog *Program) GetLog() string {
	var size C.size_t
	res := C.nvrtcGetProgramLogSize(prog.prog, &size)
	if res != C.NVRTC_SUCCESS {
		panic(res)
	}
	buf := (*C.char)(C.malloc(size))
	defer C.free(unsafe.Pointer(buf))
	res = C.nvrtcGetProgramLog(prog.prog, buf)
	if res != C.NVRTC_SUCCESS {
		panic(res)
	}
	return C.GoString(buf)
}

func (prog *Program) Compile(opts []string) {
	res := C.nvrtcCompileProgram(prog.prog, 0, (**C.char)(unsafe.Pointer(uintptr(0))))
	if res != C.NVRTC_SUCCESS {
		str := C.GoString(C.nvrtcGetErrorString(res))
		println(str)
		println(prog.GetLog())
		panic(res)
	}

	// Retrieve PTX
	var size C.size_t
	res = C.nvrtcGetPTXSize(prog.prog, &size)
	if res != C.NVRTC_SUCCESS {
		panic(res)
	}
	buf := (*C.char)(C.malloc(size))
	defer C.free(unsafe.Pointer(buf))
	res = C.nvrtcGetPTX(prog.prog, buf)

	// Store PTX into bytes
	prog.PTX = C.GoBytes(unsafe.Pointer(buf), C.int(size))
}

func (prog *Program) Destroy() {
	// TODO
}