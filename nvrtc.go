package gocudart

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda
// #include <nvrtc.h>
import "C"

func NvrtcErrorString(res C.nvrtcResult) string {
	switch res {
	case C.NVRTC_SUCCESS:
		return "NVRTC_SUCCESS"
	case C.NVRTC_ERROR_OUT_OF_MEMORY:
		return "NVRTC_ERROR_OUT_OF_MEMORY"
	case C.NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
		return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE"
	case C.NVRTC_ERROR_INVALID_INPUT:
		return "NVRTC_ERROR_INVALID_INPUT"
	case C.NVRTC_ERROR_INVALID_PROGRAM:
		return "NVRTC_ERROR_INVALID_PROGRAM"
	case C.NVRTC_ERROR_INVALID_OPTION:
		return "NVRTC_ERROR_INVALID_OPTION"
	case C.NVRTC_ERROR_COMPILATION:
		return "NVRTC_ERROR_COMPILATION"
	case C.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
		return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE"
	case C.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
		return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
	case C.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
		return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
	case C.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
		return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
	case C.NVRTC_ERROR_INTERNAL_ERROR:
		return "NVRTC_ERROR_INTERNAL_ERROR"
	default:
		return "Unknown NVRTC error string"
	}
}
