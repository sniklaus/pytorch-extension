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

	print(torch.sum(out.data - expected))

	out.backward(out.data)
# end

print('switching to DataParallel mode')

print('for me, this works with')
print('    export CUDA_VISIBLE_DEVICES="0"')
print('    export CUDA_VISIBLE_DEVICES="1"')
print('    export CUDA_VISIBLE_DEVICES="2"')
print('    export CUDA_VISIBLE_DEVICES="3"')
print('    export CUDA_VISIBLE_DEVICES="0,1"')
print('    export CUDA_VISIBLE_DEVICES="2,3"')
print('and fails with many others like')
print('    export CUDA_VISIBLE_DEVICES="0,2"')
print('    export CUDA_VISIBLE_DEVICES="0,3"')
print('    export CUDA_VISIBLE_DEVICES="1,2"')
print('    export CUDA_VISIBLE_DEVICES="1,3"')
print('    export CUDA_VISIBLE_DEVICES="0,1,2,3"')

net = torch.nn.DataParallel(Network()).cuda()
for i in range(10):
	input1 = torch.rand(64, 3, 128, 128).cuda()
	input2 = torch.rand(64, 3, 128, 128).cuda()

	expected = torch.mul(input1, input2)

	input1 = torch.autograd.Variable(input1, requires_grad=True)
	input2 = torch.autograd.Variable(input2, requires_grad=True)

	out = net(input1, input2)

	print(torch.sum(out.data - expected))

	out.backward(out.data)
# end