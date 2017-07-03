import torch

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
for i in range(10):
	input1 = torch.rand(64, 3, 128, 128).cuda()
	input2 = torch.rand(64, 3, 128, 128).cuda()

	expected = torch.mul(input1, input2)

	input1 = torch.autograd.Variable(input1, requires_grad=True)
	input2 = torch.autograd.Variable(input2, requires_grad=True)

	out = net(input1, input2)

	print(torch.sum(out.data - expected), '<-- should be 0.0')

	out.backward(out.data)
# end

print('switching to DataParallel mode')

net = torch.nn.DataParallel(Network()).cuda()
for i in range(10):
	input1 = torch.rand(64, 3, 128, 128).cuda()
	input2 = torch.rand(64, 3, 128, 128).cuda()

	expected = torch.mul(input1, input2)

	input1 = torch.autograd.Variable(input1, requires_grad=True)
	input2 = torch.autograd.Variable(input2, requires_grad=True)

	out = net(input1, input2)

	print(torch.sum(out.data - expected), '<-- should be 0.0')

	out.backward(out.data)
# end