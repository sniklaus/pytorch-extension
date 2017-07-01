int HadamardProduct_cuda_forward(
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* output
);

int HadamardProduct_cuda_backward(
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* gradOutput,
	THCudaTensor* gradInput1,
	THCudaTensor* gradInput2
);