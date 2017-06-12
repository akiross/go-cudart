package main

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda
import "C"

import (
	gocu "github.com/akiross/go-cudart"

	"fmt"
	"runtime"
	"unsafe"
)

// TODO linked list in CGO that is used inside CUDA kernel

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
}`, "vector_add"}, nil)
	pr1.Compile(nil)

	fmt.Println("Program PTX is", len(pr1.PTX), "bytes long")

	const num = 100000

	hlen := (*[1]C.int)(C.malloc(C.sizeof_int))
	ha := (*[num]C.int)(C.malloc(C.sizeof_int * num))
	hb := (*[num]C.int)(C.malloc(C.sizeof_int * num))
	hc := (*[num]C.int)(C.malloc(C.sizeof_int * num))

	hlen[0] = num
	for i := 0; i < num; i++ {
		ha[i] = C.int(i + 1)
		hb[i] = C.int(10000 - i*i)
		hc[i] = -1
	}

	mod := gocu.CreateModule()
	mod.LoadData(pr1)
	fun := mod.GetFunction("vecSum")

	dlen := gocu.NewBuffer(C.sizeof_int)
	da := gocu.NewBuffer(C.sizeof_int * num)
	db := gocu.NewBuffer(C.sizeof_int * num)
	dc := gocu.NewBuffer(C.sizeof_int * num)

	dlen.FromHost(unsafe.Pointer(&hlen[0]))
	da.FromHost(unsafe.Pointer(&ha[0]))
	db.FromHost(unsafe.Pointer(&hb[0]))

	tpb := 256
	bpg := (num + tpb - 1) / tpb
	fun.Launch1D(bpg, tpb, 0, da, db, dc, dlen)

	dc.FromDevice(unsafe.Pointer(&hc[0]))

	fmt.Println("Computation done")
	for i := 0; i < num; i++ {
		res := ha[i] + hb[i]
		if hc[i] != res {
			fmt.Println("Test failed :( res", res, "but hc", hc[i])
		}
	}
	fmt.Println("No errors, yay!")
}
