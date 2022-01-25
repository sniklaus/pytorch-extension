#!/usr/bin/env python

import torch

import hadamard # the custom layer

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super().__init__()
	# end

	def forward(self, tenOne, tenTwo):
		return hadamard.hadamard_func.apply(tenOne, tenTwo)
	# end
# end

netNetwork = Network().cuda()
for intIter in range(3):
	tenOne = torch.rand(2, 3, 8, 8).cuda()
	tenTwo = torch.rand(2, 3, 8, 8).cuda()

	tenOne = tenOne.requires_grad_()
	tenTwo = tenTwo.requires_grad_()

	tenOut = netNetwork(tenOne, tenTwo)
	tenExpected = torch.mul(tenOne, tenTwo)

	print((tenOut.data - tenExpected.data).abs().sum(), '<-- should be 0.0')
	print(torch.autograd.gradcheck(func=netNetwork, inputs=tuple([tenOne, tenTwo]), eps=0.001), '<-- should be true')
# end

print('switching to DataParallel mode')

netNetwork = torch.nn.DataParallel(Network()).cuda()
for intIter in range(3):
	tenOne = torch.rand(2, 3, 8, 8).cuda()
	tenTwo = torch.rand(2, 3, 8, 8).cuda()

	tenOne = tenOne.requires_grad_()
	tenTwo = tenTwo.requires_grad_()

	tenOut = netNetwork(tenOne, tenTwo)
	tenExpected = torch.mul(tenOne, tenTwo)

	print((tenOut.data - tenExpected.data).abs().sum(), '<-- should be 0.0')
	print(torch.autograd.gradcheck(func=netNetwork, inputs=tuple([tenOne, tenTwo]), eps=0.001), '<-- should be true')
# end