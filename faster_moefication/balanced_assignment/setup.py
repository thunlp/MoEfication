from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='balanced_assignment',
    ext_modules=[
        CUDAExtension('balanced_assignment', [
            'ba.cpp',
            # add other source files if needed
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })