#!/usr/bin/env python

import torch

import cupy

kernel_Hadamard_updateOutput = '''
	extern "C" __global__ void kernel_Hadamard_updateOutput(
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

kernel_Hadamard_updateGradInput1 = '''
	extern "C" __global__ void kernel_Hadamard_updateGradInput1(
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

kernel_Hadamard_updateGradInput2 = '''
	extern "C" __global__ void kernel_Hadamard_updateGradInput2(
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

@cupy.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
# end

class Hadamard(torch.autograd.Function):
	@staticmethod
	def forward(self, input1, input2):
		self.save_for_backward(input1, input2)

        input1 = input1.contiguous(); assert(input1.is_cuda == True)
        input2 = input2.contiguous(); assert(input2.is_cuda == True)

		output = input1.new_zeros([ input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3] ])

		if input1.is_cuda == True:
			n = output.nelement()
			cunnex('kernel_Hadamard_updateOutput')(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input1.data_ptr(), input2.data_ptr(), output.data_ptr() ]
			)

		elif input1.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	@staticmethod
	def backward(self, gradOutput):
		input1, input2 = self.saved_tensors

        gradOutput = gradOutput.contiguous(); assert(gradOutput.is_cuda == True)

		gradInput1 = input1.new_zeros([ input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3] ])
		gradInput2 = input1.new_zeros([ input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3] ])

		if input1.is_cuda == True:
			n = gradInput1.nelement()
			cunnex('kernel_Hadamard_updateGradInput1')(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input1.data_ptr(), input2.data_ptr(), gradOutput.data_ptr(), gradInput1.data_ptr(), gradInput2.data_ptr() ]
			)

			n = gradInput2.nelement()
			cunnex('kernel_Hadamard_updateGradInput2')(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input1.data_ptr(), input2.data_ptr(), gradOutput.data_ptr(), gradInput1.data_ptr(), gradInput2.data_ptr() ]
			)

		elif input1.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradInput1, gradInput2
	# end
# end