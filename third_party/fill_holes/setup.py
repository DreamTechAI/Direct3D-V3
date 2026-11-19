from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fillhole_cpp",
    ext_modules=[
        CUDAExtension(
            name="fillhole_cpp",
            sources=["src/fillhole.cpp", "src/fillhole_kernel.cu"],
            # Let PyTorch select its required C++ standard.
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
