package main

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda
// #include <cuda.h>
// void performCompute(CUfunction f, int *a, int *b, int *c);
import "C"

import (
	"fmt"
)

func Init() {
	res := C.cuInit(0)
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
}

func GetVersion() int {
	var ver C.int
	var res C.CUresult
	res = C.cuDriverGetVersion(&ver)
	if res != C.CUDA_SUCCESS {
		panic(res)
	}
	return int(ver)
}

func main() {
	Init()
	fmt.Println("CUDA Driver Version:", GetVersion())
	fmt.Println("CUDA Num devices:", GetDevicesCount())
	fmt.Println("\nCompute devices")
	devs := GetDevices()
	for i, d := range devs {
		fmt.Printf("Device %d: %s %v bytes of memory\n", i, d.Name, d.TotalMem)
	}

	// Use first device
	dev := devs[0]
	mbx, mby, mbz := dev.GetMaxBlockDim()
	fmt.Println("Max block size:", mbx, mby, mbz)
	mgx, mgy, mgz := dev.GetMaxGridDim()
	fmt.Println("Max grid size:", mgx, mgy, mgz)

	// Create context
	ctx := Create(dev, 0)
	defer ctx.Destroy() // When done
	fmt.Println("Context API version:", ctx.GetApiVersion())

	maj, min := GetNVRTCVersion()
	fmt.Println("NVRTC Version:", maj, min)

	pr1 := CreateProgram(Source{`
// Vector addition
extern "C"
__global__ void vecSum(int *a, int *b, int *c) {
	int tid = blockIdx.x;
	if (tid < 100)
		c[tid] = a[tid] + b[tid];
}`, "vector_add"}, nil)
	pr1.Compile(nil)

	fmt.Println("Program PTX is", len(pr1.PTX), "bytes long")

	ha := (*[100]C.int)(C.malloc(C.sizeof_int * 100))
	hb := (*[100]C.int)(C.malloc(C.sizeof_int * 100))
	hc := (*[100]C.int)(C.malloc(C.sizeof_int * 100))

	for i := 0; i < 100; i++ {
		ha[i] = C.int(i + 1)
		hb[i] = C.int(10000 - i*i)
		hc[i] = -1
	}

	mod := CreateModule()
	mod.LoadData(pr1)
	fun := mod.GetFunction("vecSum")

	C.performCompute(fun.id, &ha[0], &hb[0], &hc[0])

	fmt.Println("Computation done")
	for i := 0; i < 100; i++ {
		res := ha[i] + hb[i]
		if hc[i] != res {
			fmt.Println("Test failed :( res", res, "but hc", hc[i])
		}
	}
	fmt.Println("No errors, yay!")
}
