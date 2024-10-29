import torch
import torchvision.transforms as transforms
import platform
import inspect
from PIL import Image
from typing import Any, Optional


def infer_type(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def list_available_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        print(f"{num_gpus} GPU(s) available:")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def get_device(gpu_index: Optional[int] = None, _print: bool = False):
    if torch.cuda.is_available():
        if gpu_index is not None and torch.cuda.device_count() > gpu_index:
            if _print:
                print(
                    f"Device: GPU {gpu_index} [{torch.cuda.get_device_name(gpu_index)}]")
            return torch.device(f"cuda:{gpu_index}")
        else:
            if _print:
                print(f"Device: GPU 0 [{torch.cuda.get_device_name(0)}]")
            return torch.device("cuda")
    else:
        if _print:
            print(f"Device: CPU [{platform.processor()}]")
        return torch.device("cpu")


def load_image(
    path: str,
    max_size: Optional[int] = None,
    shape: Optional[tuple] = None
) -> torch.Tensor:
    image = Image.open(path)
    if shape is not None:
        image = image.resize(shape)
    elif max_size is not None:
        w, h = image.size
        larger_dim = max(w, h)
        if larger_dim > max_size:
            scale = max_size / larger_dim
            w, h = int(w * scale), int(h * scale)
            image = image.resize((w, h))
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def clamp_tensor_size(tensor: torch.Tensor, max_size: int) -> torch.Tensor:
    _, _, h, w = tensor.shape
    size = max(h, w)
    if size > max_size:
        scale = max_size / size
        new_h, new_w = int(h * scale), int(w * scale)
        resize = transforms.Resize((new_h, new_w))
        tensor = resize(tensor)
    return tensor


def is_implemented(obj, method_name):
    method = getattr(obj, method_name, None)
    if method is None:
        return False
    if callable(method):
        source = inspect.getsource(method)
        return 'raise NotImplementedError' not in source
    return True


def get_full_module_path(variable):
    variable_type = type(variable)
    module = variable_type.__module__
    class_name = variable_type.__name__
    return f"{module}.{class_name}"
