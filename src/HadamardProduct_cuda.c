#include <THC.h>
#include <THCGeneral.h>

#include "HadamardProduct_kernel.h"

extern THCState* state;

int HadamardProduct_cuda_forward(
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* output
) {
	HadamardProduct_kernel_forward(
		state,
		input1,
		input2,
		output
	);

	return 1;
}

int HadamardProduct_cuda_backward(
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* gradOutput,
	THCudaTensor* gradInput1,
	THCudaTensor* gradInput2
) {
	HadamardProduct_kernel_backward(
		state,
		input1,
		input2,
		gradOutput,
		gradInput1,
		gradInput2
	);

	return 1;
}