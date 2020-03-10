# pytorch-extension
This is an example of a CUDA extension for PyTorch which uses CuPy to compute the Hadamard product of two tensors.

For a more advanced PyTorch extension that uses CuPy as well, please see: https://github.com/szagoruyko/pyinn

## setup
Make sure to install CuPy, which can be done using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.

## usage
There is no separate build process necessary, simply run `python run.py` to test it. A minimal example of how the sample extension can be used is also shown below.

```python
import torch

import hadamard

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
	# end

	def forward(self, input1, input2):
		return hadamard.Hadamard.apply(input1, input2)
	# end
# end

net = Network().cuda()

input1 = torch.rand(64, 3, 128, 128).cuda()
input2 = torch.rand(64, 3, 128, 128).cuda()

input1 = input1.requires_grad_()
input2 = input2.requires_grad_()

output = net(input1, input2)
expected = torch.mul(input1, input2)

print(torch.sum(output.data - expected.data), '<-- should be 0.0')
```

## license
Please refer to the appropriate file within this repository.