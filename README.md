# pytorch-extension
This is an example of a CUDA extension for PyTorch which computes the Hadamard productof two tensors.

To build the extension, run `bash install.bash` and make sure that the `CUDA_HOME` environment variable is set. After succesfully building the extension, run `python test.py` to test it.  

Should you receive the error message `invalid device function` when making use of the extension, configure the CUDA architecture within `install.bash` to something your graphics card supports.