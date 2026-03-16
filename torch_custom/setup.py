import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# op-plugin 相关路径
OP_PLUGIN_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))  # op-plugin 根目录
OP_PLUGIN_INCLUDE = os.path.join(OP_PLUGIN_ROOT, "op_plugin", "include")
OP_PLUGIN_UTILS = os.path.join(OP_PLUGIN_ROOT, "op_plugin", "utils")  # 添加 utils 目录

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)

# 收集所有需要的 include 路径
include_dirs = [
    # torch_npu 相关路径
    os.path.join(PYTORCH_NPU_INSTALL_PATH, "include"),
    os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/op-plugin"),
    os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/op-plugin/op_plugin/include"),
    
    # op-plugin 相关路径
    OP_PLUGIN_INCLUDE,
    OP_PLUGIN_UTILS,
]

# 将 include_dirs 转换为编译参数
extra_compile_args = []
for dir_path in include_dirs:
    if os.path.exists(dir_path):  # 只添加存在的路径
        extra_compile_args.append('-I' + dir_path)
    else:
        print(f"Warning: Include path does not exist: {dir_path}")

exts = []
ext = NpuExtension(
    name="custom_ops_lib",
    sources=source_files,
    extra_compile_args=extra_compile_args,
)
exts.append(ext)

print("Include directories:")
for arg in extra_compile_args:
    print(f"  {arg}")

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)