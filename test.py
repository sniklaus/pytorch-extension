import torch

from HadamardProduct import HadamardProduct

class Network(torch.nn.Module):
        def __init__(self):
                super(Network, self).__init__()

        def forward(self, input1, input2):
                return HadamardProduct()(input1, input2).mean()

net = Network().cuda()
for i in range(10):
        input1 = torch.rand(64, 3, 128, 128).cuda()
        input2 = torch.rand(64, 3, 128, 128).cuda()

        input1o = torch.autograd.Variable(input1, requires_grad=True)
        input2o = torch.autograd.Variable(input2, requires_grad=True)

        input1e = torch.autograd.Variable(input1, requires_grad=True)
        input2e = torch.autograd.Variable(input2, requires_grad=True)

        output = net(input1o, input2o)
        output.backward()

        expected = torch.mul(input1e, input2e).mean()
        expected.backward()

        print('Output: \t', (output.data - expected.data).abs().sum(), '<-- should be 0.0')

        print('Gradient1: \t', (input1o.grad.data - input1e.grad.data).abs().sum(), '<-- should be 0.0')
        print('Gradient2: \t', (input2o.grad.data - input2e.grad.data).abs().sum(), '<-- should be 0.0')

        print()

print('switching to DataParallel mode')

net = torch.nn.DataParallel(Network()).cuda()
for i in range(10):
        input1 = torch.rand(64, 3, 128, 128).cuda()
        input2 = torch.rand(64, 3, 128, 128).cuda()

        input1o = torch.autograd.Variable(input1, requires_grad=True)
        input2o = torch.autograd.Variable(input2, requires_grad=True)

        input1e = torch.autograd.Variable(input1, requires_grad=True)
        input2e = torch.autograd.Variable(input2, requires_grad=True)

        output = net(input1o, input2o)
        output.backward()

        expected = torch.mul(input1e, input2e).mean()
        expected.backward()

        print('Output: \t', (output.data - expected.data).abs().sum(), '<-- should be 0.0')

        print('Gradient1: \t', (input1o.grad.data - input1e.grad.data).abs().sum(), '<-- should be 0.0')
        print('Gradient2: \t', (input2o.grad.data - input2e.grad.data).abs().sum(), '<-- should be 0.0')

        print()
