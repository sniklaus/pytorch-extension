import torch

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

from HadamardProduct import HadamardProduct

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
	# end

	def forward(self, input1, input2):
		return HadamardProduct()(input1, input2)
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

	print(torch.sum(output.data - expected.data), '<-- should be 0.0')
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

	print(torch.sum(output.data - expected.data), '<-- should be 0.0')
	print(torch.autograd.gradcheck(net, tuple([ input1, input2 ]), 0.001), '<-- should be true')
# end