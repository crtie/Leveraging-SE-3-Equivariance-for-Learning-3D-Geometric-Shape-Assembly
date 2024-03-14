"""Setup extension

Notes:
    If extra_compile_args is provided, you need to provide different instances for different extensions.
    Refer to https://github.com/pytorch/pytorch/issues/20169

"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA_ARCH can be set to your specific GPU architecture, e.g., 'sm_35', 'sm_50', 'sm_60', 'sm_70', etc.
CUDA_ARCH = None


if CUDA_ARCH is None:
    # If CUDA_ARCH is not specified, use the architecture of the current GPU.
    CUDA_ARCH = "sm_" + "75"

setup(
    name="chamfer_ext",
    ext_modules=[
        CUDAExtension(
            name="chamfer_cuda",
            sources=[
                "cuda/chamfer.cpp",
                "cuda/chamfer_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": ["-O2", "--gpu-architecture=" + CUDA_ARCH],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
