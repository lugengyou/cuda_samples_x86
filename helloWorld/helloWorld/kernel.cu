#include <stdio.h>
#include <cuda_runtime.h>

__global__ void HelloFromGPU(){
	printf("Hello world from GPU.\n");
}

void HelloFromeCPU(void) {
	printf("Hello world from CPU.\n\n");
}

void PrintDeviceProperties(void) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device %d: %s\n", dev, deviceProp.name);
	printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
	printf("Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	printf("Warp size: %d\n", deviceProp.warpSize);
	printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("Maximum number of warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize);
	printf("Maximum grid size: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Maximum block dimension: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("Maximum grid dimension: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Clock rate: %d MHz\n", deviceProp.clockRate);
	printf("Total constant memory: %lu bytes\n", deviceProp.totalConstMem);
	printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("Memory clock rate: %d MHz\n", deviceProp.memoryClockRate);
	printf("Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
	printf("Peak memory bandwidth: %f GB/s\n", 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
	printf("Device overlap: %s\n", deviceProp.deviceOverlap ? "Enabled" : "Disabled");
	printf("Kernel execution timeout: %s\n", deviceProp.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");
	printf("Integrated: %s\n", deviceProp.integrated ? ")Yes" : "No");
}

int main(void) {

	HelloFromeCPU();

	HelloFromGPU <<<1, 10 >>> ();

	PrintDeviceProperties();

	return 0;
}

