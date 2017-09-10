# pytorch-extension
This is an example of a CUDA extension for PyTorch which computes the Hadamard product of two tensors.

For the Torch version of this example extension, please see: https://github.com/sniklaus/torch-extension
<br />
For a PyTorch extension that uses CuPy instead of CFFI, please see: https://github.com/szagoruyko/pyinn

To build the extension, run `bash install.bash` and make sure that the `CUDA_HOME` environment variable is set. After successfully building the extension, run `python test.py` to test it. Should you receive an error message regarding an invalid device function when making use of the extension, configure the CUDA architecture within `install.bash` to something your graphics card supports.

```python
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

input1 = torch.rand(64, 3, 128, 128).cuda()
input2 = torch.rand(64, 3, 128, 128).cuda()

input1 = torch.autograd.Variable(input1, requires_grad=True)
input2 = torch.autograd.Variable(input2, requires_grad=True)

output = net(input1, input2)
expected = torch.mul(input1, input2)

print(torch.sum(output.data - expected.data), '<-- should be 0.0')
```