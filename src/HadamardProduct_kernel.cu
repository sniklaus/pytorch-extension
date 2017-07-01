#include <THC.h>
#include <THCGeneral.h>

#ifdef __cplusplus
	extern "C" {
#endif

__global__ void kernel_HadamardProduct_updateOutput(
	const int n,
	const float* input1,
	const float* input2,
	float* output
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	output[intIndex] = input1[intIndex] * input2[intIndex];
}

void HadamardProduct_kernel_forward(
	THCState* state,
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_HadamardProduct_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input1),
		THCudaTensor_data(state, input2),
		THCudaTensor_data(state, output)
	);

	THCudaCheck(cudaGetLastError());
}

__global__ void kernel_HadamardProduct_updateGradInput1(
	const int n,
	const float* input1,
	const float* input2,
	const float* gradOutput,
	float* gradInput1,
	float* gradInput2
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	gradInput1[intIndex] = input2[intIndex] * gradOutput[intIndex];
}

__global__ void kernel_HadamardProduct_updateGradInput2(
	const int n,
	const float* input1,
	const float* input2,
	const float* gradOutput,
	float* gradInput1,
	float* gradInput2
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	gradInput2[intIndex] = input1[intIndex] * gradOutput[intIndex];
}

void HadamardProduct_kernel_backward(
	THCState* state,
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* gradOutput,
	THCudaTensor* gradInput1,
	THCudaTensor* gradInput2
) {
	int n = 0;

	n = THCudaTensor_nElement(state, gradInput1);
	kernel_HadamardProduct_updateGradInput1<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input1),
		THCudaTensor_data(state, input2),
		THCudaTensor_data(state, gradOutput),
		THCudaTensor_data(state, gradInput1),
		THCudaTensor_data(state, gradInput2)
	);

	n = THCudaTensor_nElement(state, gradInput2);
	kernel_HadamardProduct_updateGradInput2<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input1),
		THCudaTensor_data(state, input2),
		THCudaTensor_data(state, gradOutput),
		THCudaTensor_data(state, gradInput1),
		THCudaTensor_data(state, gradInput2)
	);

	THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
	}
#endif