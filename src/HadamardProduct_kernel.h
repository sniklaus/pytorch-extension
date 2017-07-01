#ifdef __cplusplus
	extern "C" {
#endif

void HadamardProduct_kernel_forward(
	THCState* state,
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* output
);

void HadamardProduct_kernel_backward(
	THCState* state,
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* gradOutput,
	THCudaTensor* gradInput1,
	THCudaTensor* gradInput2
);

#ifdef __cplusplus
	}
#endif