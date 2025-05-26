#!/usr/bin/env python

import collections
import cupy
import os
import torch


##########################################################


def cuda_int32(intIn:int):
    return cupy.int32(intIn)
# end


def cuda_float32(fltIn:float):
    return cupy.float32(fltIn)
# end


@cupy.memoize(for_each_device=True)
def cuda_launch(strFunction:str, strKernel:str):
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = cupy.cuda.get_cuda_path()
    # end

    return cupy.RawKernel(strKernel, strFunction)
# end


class hadamard_func(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(self, tenOne, tenTwo):
        tenOne = tenOne.float() # manual cast in case amp is not enabled
        tenTwo = tenTwo.float() # manual cast in case amp is not enabled

        tenOne = tenOne.contiguous(); assert(tenOne.is_cuda == True)
        tenTwo = tenTwo.contiguous(); assert(tenTwo.is_cuda == True)

        tenOut = tenOne.new_zeros([tenOne.shape[0], tenOne.shape[1], tenOne.shape[2], tenOne.shape[3]])

        if tenOne.is_cuda == True:
            cuda_launch('hadamard_out', '''
                extern "C" __global__ void __launch_bounds__(512) hadamard_out(
                    const int n,
                    const float* __restrict__ tenOne,
                    const float* __restrict__ tenTwo,
                    float* __restrict__ tenOut
                ) {
                    int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

                    if (intIndex >= n) {
                        return;
                    }

                    tenOut[intIndex] = tenOne[intIndex] * tenTwo[intIndex];
                }
            ''')(
                grid=tuple([int((tenOut.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cupy.int32(tenOut.nelement()), tenOne.data_ptr(), tenTwo.data_ptr(), tenOut.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )

        elif tenOne.is_cuda == False:
            raise NotImplementedError()

        # end

        self.save_for_backward(tenOne, tenTwo)

        return tenOut
    # end

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(self, tenOutgrad):
        tenOne, tenTwo = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous(); assert(tenOutgrad.is_cuda == True)

        tenOnegrad = tenOne.new_zeros([tenOne.shape[0], tenOne.shape[1], tenOne.shape[2], tenOne.shape[3]])
        tenTwograd = tenOne.new_zeros([tenOne.shape[0], tenOne.shape[1], tenOne.shape[2], tenOne.shape[3]])

        if tenOne.is_cuda == True:
            cuda_launch('hadamard_onegrad', '''
                extern "C" __global__ void __launch_bounds__(512) hadamard_onegrad(
                    const int n,
                    const float* __restrict__ tenOne,
                    const float* __restrict__ tenTwo,
                    const float* __restrict__ tenOutgrad,
                    float* __restrict__ tenOnegrad,
                    float* __restrict__ tenTwograd
                ) {
                    int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

                    if (intIndex >= n) {
                        return;
                    }

                    tenOnegrad[intIndex] = tenTwo[intIndex] * tenOutgrad[intIndex];
                }
            ''')(
                grid=tuple([int((tenOnegrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cupy.int32(tenOnegrad.nelement()), tenOne.data_ptr(), tenTwo.data_ptr(), tenOutgrad.data_ptr(), tenOnegrad.data_ptr(), tenTwograd.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )

            cuda_launch('hadamard_twograd', '''
                extern "C" __global__ void __launch_bounds__(512) hadamard_twograd(
                    const int n,
                    const float* __restrict__ tenOne,
                    const float* __restrict__ tenTwo,
                    const float* __restrict__ tenOutgrad,
                    float* __restrict__ tenOnegrad,
                    float* __restrict__ tenTwograd
                ) {
                    int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

                    if (intIndex >= n) {
                        return;
                    }

                    tenTwograd[intIndex] = tenOne[intIndex] * tenOutgrad[intIndex];
                }
            ''')(
                grid=tuple([int((tenTwograd.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cupy.int32(tenTwograd.nelement()), tenOne.data_ptr(), tenTwo.data_ptr(), tenOutgrad.data_ptr(), tenOnegrad.data_ptr(), tenTwograd.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )

        elif tenOne.is_cuda == False:
            raise NotImplementedError()

        # end

        return tenOnegrad, tenTwograd
    # end
# end
