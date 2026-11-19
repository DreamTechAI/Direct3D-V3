from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cudavoxelize",
    ext_modules=[
        CUDAExtension("cudavoxelize", ["src/voxel_kernel.cu", "src/voxel_cuda.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
