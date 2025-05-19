import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch

def make_cuda_ext(name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

setup(
    name='rctrans_voxel_pooling',
    version='0.1',
    description='RCTrans voxel pooling CUDA extensions',
    packages=find_packages(),
    ext_modules=[
        make_cuda_ext(
            name='voxel_pooling_inference_ext',
            module='projects.mmdet3d_plugin.models.utils.voxel_pooling_inference',
            sources=['src/voxel_pooling_inference_forward.cpp'],
            sources_cuda=['src/voxel_pooling_inference_forward_cuda.cu'],
        ),
        make_cuda_ext(
            name='voxel_pooling_train_ext',
            module='projects.mmdet3d_plugin.models.utils.voxel_pooling_train',
            sources=['src/voxel_pooling_train_forward.cpp'],
            sources_cuda=['src/voxel_pooling_train_forward_cuda.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
