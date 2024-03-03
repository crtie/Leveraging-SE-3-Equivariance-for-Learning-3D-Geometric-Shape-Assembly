"""Setup extension

Notes:
    If extra_compile_args is provided, you need to provide different instances for different extensions.
    Refer to https://github.com/pytorch/pytorch/issues/20169

"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Ensure the CUDA version is set to 10.2
CUDA_VERSION = '10.2'

setup(
    name='chamfer_ext',
    ext_modules=[
        CUDAExtension(
            name='chamfer_cuda',
            sources=[
                'cuda/chamfer.cpp',
                'cuda/chamfer_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-g'], # Use C++14 standard for C++ compilation
                'nvcc': [
                    '-O2', 
                    '-ccbin', '/opt/ohpc/pub/gcc/gcc-7.5.0/bin/gcc',  # Specify the path to the GCC executable
                    '-gencode', 'arch=compute_75,code=sm_75'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

