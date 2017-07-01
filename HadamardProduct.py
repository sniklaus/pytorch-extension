import torch

import _ext.cunnex

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
			_ext.cunnex.HadamardProduct_cuda_forward(
				input1,
				input2,
				output
			)

		elif input1.is_cuda == False:
			assert(False) # CPU VERSION NOT IMPLEMENTED

		# end

		return output
	# end

	def backward(self, gradOutput):
		input1, input2 = self.saved_tensors

		assert(gradOutput.is_contiguous() == True)

		gradInput1 = input1.new().resize_as_(input1).zero_()
		gradInput2 = input1.new().resize_as_(input1).zero_()

		if input1.is_cuda == True:
			_ext.cunnex.HadamardProduct_cuda_backward(
				input1,
				input2,
				gradOutput,
				gradInput1,
				gradInput2
			)

		elif input1.is_cuda == False:
			assert(False) # CPU VERSION NOT IMPLEMENTED

		# end

		return gradInput1, gradInput2
	# end
# end