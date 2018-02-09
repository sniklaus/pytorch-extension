import torch

import cupy

kernel_HadamardProduct_updateOutput = '''
	extern "C" __global__ void kernel_HadamardProduct_updateOutput(
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
'''

kernel_HadamardProduct_updateGradInput1 = '''
	extern "C" __global__ void kernel_HadamardProduct_updateGradInput1(
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
'''

kernel_HadamardProduct_updateGradInput2 = '''
	extern "C" __global__ void kernel_HadamardProduct_updateGradInput2(
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
'''

@cupy.util.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
# end

class HadamardProduct(torch.autograd.Function):
	def __init__(self):
		super(HadamardProduct, self).__init__()
	# end

	def forward(self, input1, input2):
		self.save_for_backward(input1, input2)

		assert(input1.is_contiguous() == True)
		assert(input2.is_contiguous() == True)

		output = input1.new().resize_as_(input1).zero_()

		if input1.is_cuda == True:
			class Stream:
				ptr=torch.cuda.current_stream().cuda_stream
			# end

			n = output.nelement()
			cunnex('kernel_HadamardProduct_updateOutput')(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input1.data_ptr(), input2.data_ptr(), output.data_ptr() ],
				stream=Stream
			)

		elif input1.is_cuda == False:
			raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

		# end

		return output
	# end

	def backward(self, gradOutput):
		input1, input2 = self.saved_tensors

		assert(gradOutput.is_contiguous() == True)

		gradInput1 = input1.new().resize_as_(input1).zero_()
		gradInput2 = input1.new().resize_as_(input1).zero_()

		if input1.is_cuda == True:
			class Stream:
				ptr=torch.cuda.current_stream().cuda_stream
			# end

			n = gradInput1.nelement()
			cunnex('kernel_HadamardProduct_updateGradInput1')(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input1.data_ptr(), input2.data_ptr(), gradOutput.data_ptr(), gradInput1.data_ptr(), gradInput2.data_ptr() ],
				stream=Stream
			)

			n = gradInput2.nelement()
			cunnex('kernel_HadamardProduct_updateGradInput2')(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input1.data_ptr(), input2.data_ptr(), gradOutput.data_ptr(), gradInput1.data_ptr(), gradInput2.data_ptr() ],
				stream=Stream
			)

		elif input1.is_cuda == False:
			raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

		# end

		return gradInput1, gradInput2
	# end
# end