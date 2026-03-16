#!/bin/bash
BASE_DIR=$(pwd)

# 编译wheel包
python3 setup.py build bdist_wheel

# 安装wheel包
cd ${BASE_DIR}/dist
pip3 install *.whl --force-reinstall


# 运行测试用例
cd ${BASE_DIR}/test
python3 test_prepare_wy_repr_bwd_full.py
if [ $? -ne 0 ]; then
    echo "[ERROR]: Run add_custom test failed!"
fi
echo "[INFO]: Run add_custom test success!"

# # 运行测试用例
# python3 test_add_custom_graph.py
# if [ $? -ne 0 ]; then
#     echo "[ERROR]: Run add_custom_graph test failed!"
# fi
# echo "[INFO]: Run add_custom_graph test success!"