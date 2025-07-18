import pathlib
import pybind11

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define CPP/CUDA extensions for the checkpointing engine.
ckpt_engine_path = "src/"

abs_ckpt_engine_path = pathlib.Path(f"{pathlib.Path(__file__).parent.resolve()}/{ckpt_engine_path}")
pybind11_include_path = pybind11.get_include()

extensions = [
    CUDAExtension(
        name='datastates_engine',
        sources=[
            f'{ckpt_engine_path}/pool/mem_pool.cpp',
            f'{ckpt_engine_path}/tiers/gpu_tier.cpp',
            f'{ckpt_engine_path}/tiers/host_tier.cpp',
            f'{ckpt_engine_path}/engine.cpp',
            f'{ckpt_engine_path}/py_datastates_llm.cpp'
        ],
        include_dirs=[
            f"{abs_ckpt_engine_path}/common",
            f"{abs_ckpt_engine_path}/pool",
            f"{abs_ckpt_engine_path}/tiers",
            f"{abs_ckpt_engine_path}",
            pybind11_include_path
        ]
    )
]

setup(
    name="datastates",
    version="0.0.1",
    author="ANL",
    packages=find_packages(include=['datastates', 'datastates.*']),
    include_package_data=True,
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension
    },
    description="Datastates-LLM checkpointing engine",
    long_description=open("README.md").read() if pathlib.Path("README.md").exists() else "",
    install_requires=["pybind11", "torch"],
)