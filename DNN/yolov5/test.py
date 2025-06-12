import torch
import os
from pathlib import Path
import pickle
import sys


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'PosixPath':
            return Path
        return super().find_class(module, name)
class CustomPickleModule:
    Unpickler = CustomUnpickler
    load = pickle.load
    loads = pickle.loads
    dump = pickle.dump
    dumps = pickle.dumps

weights_path = "DNN/yolov5/pre_train_yolov5/coco5k_Almost_Natural_Light_100/weights/best.pt"
state_dict = torch.load(weights_path, map_location='cpu', pickle_module=CustomPickleModule)


print("模型文件中的所有键：")
for key in state_dict.keys():
    print(f"键名: {key}")
    
    # 如果是字典，打印其内容
    if isinstance(state_dict[key], dict):
        print("  内容:")
        for sub_key in state_dict[key].keys():
            print(f"    {sub_key}: {type(state_dict[key][sub_key])}")
            # 如果值是路径对象，打印出来
            if str(type(state_dict[key][sub_key])).find('PosixPath') != -1:
                print(f"    发现 PosixPath: {state_dict[key][sub_key]}")
    print("---")

# 保存有问题的路径信息
problematic_paths = []
for key, value in state_dict.items():
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            if str(type(sub_value)).find('PosixPath') != -1:
                problematic_paths.append((key, sub_key, sub_value))

print("\n发现的所有 PosixPath：")
for key, sub_key, path in problematic_paths:
    print(f"在 {key}.{sub_key} 中发现: {path}")