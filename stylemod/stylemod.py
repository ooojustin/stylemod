import torch
from stylemod.core.factory import ModelFactory
from stylemod.core.base import BaseModel
from stylemod.core.cnn import CNNBaseModel
from stylemod.core.transformer import TransformerBaseModel
from stylemod.models import Model
from stylemod import utils
from tqdm import tqdm
from typing import Union, Optional, Literal
from torch.optim.adam import Adam
from torch.optim.lbfgs import LBFGS
from PIL import Image


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
    return_type: Literal["tensor", "pil"] = "tensor",
    _print: bool = True
) -> Union[torch.Tensor, Image.Image]:
    if isinstance(model, Model):
        model = ModelFactory.create(model.name)
    elif isinstance(model, BaseModel):
        model = model
    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. Must be either a `Model` enum or a superclass of `BaseModel`.")

    device = utils.get_device(gpu_index, _print=_print)
    model.set_device(device)
    if model.eval_mode:
        model.eval()

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
        raise TypeError(
            f"Invalid type for content_image:  {type(content_image)}. Must be either a str or torch.Tensor.")

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
        raise TypeError(
            f"Invalid type for style_image:  {type(style_image)}. Must be either a str or torch.Tensor.")

    if model.normalization is not None:
        content = model.normalize_tensor(content)
        style = model.normalize_tensor(style)

    content_features = model.get_features(
        content, layers=[model.content_layer])
    style_features = model.get_features(style, layers=model.style_layers)

    if isinstance(model, TransformerBaseModel):
        model.compute_style_attention(style)

    target = content.clone().requires_grad_(True).to(device)

    if optimizer_type == "lbfgs":
        optimizer = LBFGS([target], max_iter=steps, lr=learning_rate)
    elif optimizer_type == "adam":
        optimizer = Adam([target], lr=learning_rate)

    def loss_step():
        target_features = model.get_features(
            target, layers=[model.content_layer] + model.style_layers
        )
        content_loss = torch.mean(
            (target_features[model.content_layer] -
             content_features[model.content_layer]) ** 2
        )
        if isinstance(model, CNNBaseModel):
            style_loss = model.calc_style_loss(
                target_features, style_features, target.device)
        elif isinstance(model, TransformerBaseModel):
            style_loss = model.calc_style_loss(target)
        else:
            raise AssertionError("Invalid model.")
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward(retain_graph=model.retain_graph)
        return total_loss

    step_range = tqdm(
        range(steps), desc="Loss Calculation") if _print else range(steps)
    for step in step_range:
        if isinstance(optimizer, Adam):
            optimizer.zero_grad()
            total_loss = loss_step()
            optimizer.step()
        elif isinstance(optimizer, LBFGS):
            total_loss = optimizer.step(loss_step)
        else:
            raise AssertionError("Invalid optimizer.")

        if step % 10 == 0 and isinstance(step_range, tqdm):
            step_range.set_postfix({'total_loss': total_loss.item()})

    tensor = target.clone().cpu().detach()
    if return_type == "pil":
        if model.normalization is not None:
            tensor = model.denormalize_tensor(tensor)
        tensor = tensor.clamp(0, 1).squeeze().permute(1, 2, 0)
        arr = (tensor.numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr)
        return pil
    else:
        return tensor