import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """未切分参数的默认加载路径：整块复制 checkpoint tensor。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """流式加载 safetensors，并把 checkpoint 名称映射到融合/TP 参数。

    例如 checkpoint 中独立的 ``q_proj``、``k_proj``、``v_proj`` 会依次写入
    本地融合参数 ``qkv_proj`` 的不同区间，避免先拼完整权重再切分。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # shard_id 可以是 "q"/"k"/"v"，也可以是 gate/up 的 0/1。
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
