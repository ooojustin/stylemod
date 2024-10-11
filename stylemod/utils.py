import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional


def list_available_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        print(f"{num_gpus} GPU(s) available:")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def get_device(gpu_index=None):
    if torch.cuda.is_available():
        if gpu_index is not None and torch.cuda.device_count() > gpu_index:
            print(
                f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
            return torch.device(f"cuda:{gpu_index}")
        else:
            print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def load_image(
    path: str,
    max_size: Optional[int] = None,
    shape: Optional[tuple] = None
) -> torch.Tensor:
    image = Image.open(path)

    if shape is not None:
        # resize the image to the provided shape
        image = image.resize(shape)
    elif max_size is not None:
        # resize while perserving aspect ratio
        original_width, original_height = image.size
        largest_dim = max(original_width, original_height)
        if largest_dim > max_size:
            scale_factor = max_size / largest_dim
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = image.resize((new_width, new_height))

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


def get_full_module_path(variable):
    variable_type = type(variable)
    module = variable_type.__module__
    class_name = variable_type.__name__
    return f"{module}.{class_name}"
