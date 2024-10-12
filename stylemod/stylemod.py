import torch
from stylemod.core.factory import ModelFactory
from stylemod.core.base import BaseModel
from stylemod.core.cnn import CNNBaseModel
from stylemod.core.transformer import TransformerBaseModel
from stylemod.models import Model
from stylemod import utils
from torch.optim.adam import Adam
from torch.optim.lbfgs import LBFGS
from PIL import Image
from typing import Union, Optional, Literal


def style_transfer(
    content_image: Union[str, torch.Tensor],
    style_image: Union[str, torch.Tensor],
    model: Union[Model, BaseModel] = Model.VGG19,
    max_size: Optional[int] = None,
    steps: int = 1000,
    gpu_index: Optional[int] = None,
    content_weight: float = 1e4,
    style_weight: float = 1e2,
    learning_rate: float = 0.003,
    optimizer_type: Literal["adam", "lbfgs"] = "adam",
    return_type: Literal["tensor", "pil"] = "tensor"
) -> Union[torch.Tensor, Image.Image]:
    if isinstance(model, Model):
        model_instance = ModelFactory.create(model.name)
    elif isinstance(model, BaseModel):
        model_instance = model
    else:
        raise ValueError(
            f"Unsupported model type: {type(model)}. Must be either a `Model` enum or a `BaseModel` instance.")

    device = utils.get_device(gpu_index)
    model_instance.set_device(device)
    if model_instance.eval_mode:
        model_instance.eval()

    content_layer = model_instance.content_layer
    style_layers = model_instance.style_layers

    if isinstance(content_image, str):
        content = utils.load_image(
            path=content_image,
            max_size=max_size
        ).to(device)
    elif isinstance(content_image, torch.Tensor):
        content = content_image.to(device)
        if max_size is not None:
            content = utils.clamp_tensor_size(content, max_size)
    else:
        raise ValueError(
            f"Invalid type for content_image: expected str or torch.Tensor, but got {type(content_image)}.")

    if isinstance(style_image, str):
        style = utils.load_image(
            path=style_image,
            shape=content.shape[-2:],
            max_size=max_size
        ).to(device)
    elif isinstance(style_image, torch.Tensor):
        style = style_image.to(device)
        if max_size is not None:
            style = utils.clamp_tensor_size(style, max_size)
    else:
        raise ValueError(
            f"Invalid type for style_image: expected str or torch.Tensor, but got {type(style_image)}.")

    if model_instance.normalization is not None:
        content = model_instance.normalize_tensor(content)
        style = model_instance.normalize_tensor(style)

    content_features = model_instance.get_features(
        content, layers=[content_layer])
    style_features = model_instance.get_features(style, layers=style_layers)

    target = content.clone().requires_grad_(True).to(device)

    # precompute style loss if transformer model
    if isinstance(model_instance, TransformerBaseModel):
        model_instance.precompute_style_attention(style)

    if optimizer_type == "lbfgs":
        optimizer = LBFGS([target], max_iter=steps, lr=learning_rate)
    elif optimizer_type == "adam":
        optimizer = Adam([target], lr=learning_rate)

    def train():
        target_features = model_instance.get_features(
            target, layers=[content_layer] + style_layers
        )

        content_loss = torch.mean(
            (target_features[content_layer] -
             content_features[content_layer]) ** 2
        )

        style_loss = 0
        if isinstance(model_instance, CNNBaseModel):
            style_loss = model_instance.get_style_loss(
                target_features, style_features, target.device)
        elif isinstance(model_instance, TransformerBaseModel):
            style_loss = model_instance.get_style_loss(target)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward(retain_graph=model_instance.retain_graph)

        return total_loss

    for step in range(steps):

        total_loss = torch.zeros(0)
        if isinstance(optimizer, Adam):
            optimizer.zero_grad()
            total_loss = train()
            optimizer.step()
        elif isinstance(optimizer, LBFGS):
            total_loss = optimizer.step(train)

        if step % 10 == 0:
            print(f"Step {step}, total loss: {total_loss.item()}")

    output_tensor = target.clone().cpu().detach()
    print(f"Style transfer complete!")

    if return_type == "pil":

        if model_instance.normalization is not None:
            output_tensor = model_instance.denormalize_tensor(output_tensor)

        # permutation: [channels, height, width] -> [height, width, channels]
        output_tensor = output_tensor.clamp(0, 1).squeeze()
        output_tensor = output_tensor.permute(1, 2, 0)
        output_numpy = (output_tensor.numpy() * 255).astype("uint8")
        output_pil = Image.fromarray(output_numpy)
        return output_pil

    return output_tensor
