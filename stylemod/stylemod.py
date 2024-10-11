import torch
from stylemod.core.factory import ModelFactory
from stylemod.core.base_model import BaseModel
from stylemod.models import Model
from stylemod import utils
from torch.optim.adam import Adam
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
    style_weights = model_instance.style_weights

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

    content_loss = torch.mean(
        (content_features[content_layer] - content_features[content_layer]) ** 2)
    style_grams = {layer: model_instance.gram_matrix(
        style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    optimizer = Adam([target], lr=learning_rate)

    for step in range(steps):
        target_features = model_instance.get_features(
            target, layers=[content_layer] + list(style_layers))

        content_loss = torch.mean(
            (target_features[content_layer] - content_features[content_layer]) ** 2)

        style_loss = 0
        for layer in style_layers:
            target_gram = model_instance.gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += style_weights[layer] * \
                torch.mean((target_gram - style_gram) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=model_instance.retain_graph)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, total loss: {total_loss.item()}")

    output_tensor = target.clone().cpu().detach()
    print(f"Style transfer complete!")

    if return_type == "pil":

        if model_instance.normalization is not None:
            mean, std = model_instance.normalization
            for t, m, s in zip(output_tensor, mean, std):
                t.mul_(s).add_(m)

        # permutation: [channels, height, width] -> [height, width, channels]
        output_tensor = output_tensor.clamp(0, 1).squeeze()
        output_tensor = output_tensor.permute(1, 2, 0)
        output_numpy = (output_tensor.numpy() * 255).astype("uint8")
        output_pil = Image.fromarray(output_numpy)
        return output_pil

    return output_tensor
