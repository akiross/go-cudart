package main

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda
// #include <cuda.h>
// void performCompute(CUfunction f, int *a, int *b, int *c, int len);
/*
void performCompute(void *f, int *ha, int *hb, int *hc, int len) {
	CUfunction func = *(CUfunction*)f;
	// Create stream
	CUstream stream = 0; // Default stream
	//errorCheck(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "Error while creating stream: %d\n");

	// Setup device memory
	CUdeviceptr da, db, dc;
	errorCheck(cuMemAlloc(&da, sizeof(int) * len), "Error while allocating da: %d\n");
	errorCheck(cuMemAlloc(&db, sizeof(int) * len), "Error while allocating db: %d\n");
	errorCheck(cuMemAlloc(&dc, sizeof(int) * len), "Error while allocating dc: %d\n");

	// Copy from host to device
	errorCheck(cuMemcpyHtoD(da, ha, sizeof(int) * len), "Error while copying ha->da: %d\n");
	errorCheck(cuMemcpyHtoD(db, hb, sizeof(int) * len), "Error while copying hb->db: %d\n");

	void *args[] = {&da, &db, &dc, &len};
	int tpb = 256;
	int bpg = (len + tpb - 1) / tpb;
	errorCheck(cuLaunchKernel(func,
				bpg, 1, 1,
				tpb, 1, 1,
				0,
				stream,
				args, 0), "Error while launching kernel: %d\n");

	errorCheck(cuMemcpyDtoH(hc, dc, sizeof(int) * len), "Error while copying back: %d\n");
	printf("Finished compute of %d elements\n", len);
}
*/
import "C"

import (
	gocu "github.com/akiross/go-cudart"

	"fmt"
	"runtime"
	"unsafe"
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

	ha := (*[num]C.int)(C.malloc(C.sizeof_int * num))
	hb := (*[num]C.int)(C.malloc(C.sizeof_int * num))
	hc := (*[num]C.int)(C.malloc(C.sizeof_int * num))

	for i := 0; i < num; i++ {
		ha[i] = C.int(i + 1)
		hb[i] = C.int(10000 - i*i)
		hc[i] = -1
	}

	mod := gocu.CreateModule()
	mod.LoadData(pr1)
	fun := mod.GetFunction("vecSum")

	C.performCompute(unsafe.Pointer(&fun.Id), &ha[0], &hb[0], &hc[0], C.int(num))

	fmt.Println("Computation done")
	for i := 0; i < num; i++ {
		res := ha[i] + hb[i]
		if hc[i] != res {
			fmt.Println("Test failed :( res", res, "but hc", hc[i])
		}
	}
	fmt.Println("No errors, yay!")
}
