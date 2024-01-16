# pytorch-extension
This is an example of a CUDA extension/function/layer for PyTorch which uses CuPy to compute the Hadamard product of two tensors.

For a more advanced extension that uses CuPy as well, please see: https://github.com/sniklaus/softmax-splatting
<br />
And yet another extension that uses CuPy can be found here: https://github.com/sniklaus/revisiting-sepconv

## setup
Make sure to install CuPy, which can be done using `pip install cupy` or alternatively using one of the provided [binary packages](https://docs.cupy.dev/en/stable/install.html#installing-cupy) as outlined in the CuPy repository.

## usage
There is no separate build process necessary, simply run `python run.py` to test it. A minimal example of how the sample extension can be used is also shown below.

```python
import torch

import hadamard # the custom layer

class Network(torch.nn.Module):
	def __init__(self):
		super().__init__()
	# end

	def forward(self, tenOne, tenTwo):
		return hadamard.hadamard_func.apply(tenOne, tenTwo)
	# end
# end

netNetwork = Network().cuda()

tenOne = torch.rand(size=[64, 3, 128, 128], dtype=torch.float32, device=torch.device('cuda')).requires_grad_()
tenTwo = torch.rand(size=[64, 3, 128, 128], dtype=torch.float32, device=torch.device('cuda')).requires_grad_()

tenOut = netNetwork(tenOne, tenTwo)
tenExpected = torch.mul(tenOne, tenTwo)

print(torch.sum(tenOut.data - tenExpected.data), '<-- should be 0.0')
```

## license
Please refer to the appropriate file within this repository.