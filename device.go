package gocudart

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda -lnvrtc
// #include <cuda.h>
// #include <nvrtc.h>
import "C"

type Device struct {
	id C.CUdevice

	Name     string
	TotalMem uint
}

func (dev *Device) GetAttribute(attrib C.CUdevice_attribute) int {
	var val C.int
	res := C.cuDeviceGetAttribute(&val, attrib, dev.id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return int(val)
}

func (dev *Device) GetMaxBlockDim() (x, y, z int) {
	x = dev.GetAttribute(C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
	y = dev.GetAttribute(C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
	z = dev.GetAttribute(C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
	return
}

func (dev *Device) GetMaxGridDim() (x, y, z int) {
	x = dev.GetAttribute(C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
	y = dev.GetAttribute(C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
	z = dev.GetAttribute(C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
	return
}

func GetDevicesCount() int {
	var num C.int
	var res C.CUresult
	res = C.cuDeviceGetCount(&num)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	return int(num)
}

func GetDevice(ordinal int) *Device {
	var dev Device
	res := C.cuDeviceGet(&dev.id, C.int(ordinal))
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	// Get device name
	var str = C.malloc(1024)
	defer C.free(str)
	res = C.cuDeviceGetName((*C.char)(str), 1024, dev.id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	dev.Name = C.GoString((*C.char)(str))
	// Get device total memory
	var totMem C.size_t
	res = C.cuDeviceTotalMem(&totMem, dev.id)
	if res != C.CUDA_SUCCESS {
		panic(CudaErrorString(res))
	}
	dev.TotalMem = uint(totMem)
	return &dev
}

func GetDevices() []*Device {
	numDevs := GetDevicesCount()
	devs := make([]*Device, numDevs)
	for i := 0; i < numDevs; i++ {
		devs[i] = GetDevice(i)
	}
	return devs
}
