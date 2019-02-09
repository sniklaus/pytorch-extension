import torch

import hadamard

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.0

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
	# end

	def forward(self, input1, input2):
		return hadamard.Hadamard()(input1, input2)
	# end
# end

net = Network().cuda()
for i in range(3):
	input1 = torch.rand(2, 3, 8, 8).cuda()
	input2 = torch.rand(2, 3, 8, 8).cuda()

	input1 = input1.requires_grad_()
	input2 = input2.requires_grad_()

	output = net(input1, input2)
	expected = torch.mul(input1, input2)

	print((output.data - expected.data).abs().sum(), '<-- should be 0.0')
	print(torch.autograd.gradcheck(net, tuple([ input1, input2 ]), 0.001), '<-- should be true')
# end

print('switching to DataParallel mode')

net = torch.nn.DataParallel(Network()).cuda()
for i in range(3):
	input1 = torch.rand(2, 3, 8, 8).cuda()
	input2 = torch.rand(2, 3, 8, 8).cuda()

	input1 = input1.requires_grad_()
	input2 = input2.requires_grad_()

	output = net(input1, input2)
	expected = torch.mul(input1, input2)

	print((output.data - expected.data).abs().sum(), '<-- should be 0.0')
	print(torch.autograd.gradcheck(net, tuple([ input1, input2 ]), 0.001), '<-- should be true')
# end