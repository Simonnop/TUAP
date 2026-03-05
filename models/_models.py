from pathlib import Path
import importlib

# 将自动加载models文件夹下所有时序模型 模型名称与文件名称相同 △文件名不要以_开头否则会被忽略
folder_path = Path('models')
model_dict={}
for file_path in folder_path.glob('[!_]*.py'):
    module_name = file_path.stem
    # print(module_name)
    try:
        # 动态导入模块
        module = importlib.import_module(str(file_path)[:-3].replace('/','.'))
        # 将模块添加到字典中
        model_dict[module_name] = module
    except ImportError:
        print(f"Failed to import {module_name}")
