import torch
from torchvision import transforms
from PIL import Image


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


def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path)

    # resize the image to either the shape or max_size
    if shape is not None:
        image = image.resize(shape)
    else:
        size = max_size if max(image.size) > max_size else max(image.size)
        image = image.resize((size, int(size * image.size[1] / image.size[0])))

    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image


def get_full_module_path(variable):
    variable_type = type(variable)
    module = variable_type.__module__
    class_name = variable_type.__name__
    return f"{module}.{class_name}"
