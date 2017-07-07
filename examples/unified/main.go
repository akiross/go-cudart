package main

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda
import "C"

import (
	gocu "github.com/akiross/go-cudart"

	"fmt"
	"runtime"
	//"unsafe"
)

func main() {
	gocu.Init()
	fmt.Println("CUDA Driver Version:", gocu.GetVersion())
	fmt.Println("CUDA Num devices:", gocu.GetDevicesCount())
	fmt.Println("\nCompute devices")
	devs := gocu.GetDevices()
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
	runtime.LockOSThread()
	ctx := gocu.Create(dev, 0)
	defer ctx.Destroy() // When done
	fmt.Println("Context API version:", ctx.GetApiVersion())

	// Force GC to check if context is safe
	runtime.GC()

	maj, min := gocu.GetNVRTCVersion()
	fmt.Println("NVRTC Version:", maj, min)

	pr1 := gocu.CreateProgram(gocu.Source{`
// Vector addition
extern "C"
__global__ void vecSum(int *a, int *b, int *c, int len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len)
		c[tid] = a[tid] + b[tid];
}`, "vector_add"})
	pr1.Compile()

	fmt.Println("Program PTX is", len(pr1.PTX), "bytes long")

	const num = 100000

	// Unified memory
	abuf, a := gocu.AllocManagedInt32(num * C.sizeof_int)
	bbuf, b := gocu.AllocManagedInt32(num * C.sizeof_int)
	cbuf, c := gocu.AllocManagedInt32(num * C.sizeof_int)

	//hlen[0] = num
	for i := 0; i < num; i++ {
		a[i] = int32(i + 1)
		b[i] = int32(10000 - i*i)
		c[i] = -1
	}

	mod := gocu.CreateModule()
	mod.LoadData(pr1)
	fun := mod.GetFunction("vecSum")

	dlen := gocu.NewBuffer(C.sizeof_int)
	dlen.FromIntC(num)

	tpb := 256
	bpg := (num + tpb - 1) / tpb
	fun.Launch1D(bpg, tpb, 0, abuf, bbuf, cbuf, dlen)

	ctx.Synchronize() // Copy back the data

	fmt.Println("Computation done")
	for i := 0; i < num; i++ {
		res := a[i] + b[i]
		if c[i] != res {
			fmt.Println("Test failed :( res", res, "but c", c[i])
		}
	}
	fmt.Println("No errors, yay!")
}
