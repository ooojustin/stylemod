import torch
from stylemod.core.factory import ModelFactory
from stylemod.models import Model
from stylemod import utils
from torch.optim.adam import Adam
from PIL import Image
from typing import Union, Optional


def style_transfer(content_image: str, style_image: str, output_image: str, steps: int, max_size: int, model: Union[str, Model], gpu_index: Optional[int] = None):
    """
    Core function to perform style transfer.

    Args:
        content_image (str): Path to the content image.
        style_image (str): Path to the style image.
        output_image (str): Output filename for the stylized image.
        steps (int): Number of optimization steps.
        max_size (int): Maximum size of input images.
        model (Union[str, Model]): Model to use for feature extraction (string or Model enum).
        gpu_index (int, optional): GPU index to use. Defaults to None (use CPU if no GPU).
    """
    utils.list_available_gpus()
    device = utils.get_device(gpu_index)

    model_instance = ModelFactory.create(model)
    model_instance.set_device(device)
    if model_instance.eval_mode:
        model_instance.eval()

    print("Model:", model_instance.name)

    content_layer = model_instance.content_layer
    style_layers = model_instance.style_layers
    style_weights = model_instance.style_weights

    content = utils.load_image(content_image, max_size=max_size).to(device)
    style = utils.load_image(
        style_image, shape=content.shape[-2:], max_size=max_size).to(device)

    content_features = model_instance.get_features(
        content, layers=[content_layer])
    style_features = model_instance.get_features(style, layers=style_layers)

    content_loss = torch.mean(
        (content_features[content_layer] - content_features[content_layer]) ** 2)
    style_grams = {layer: model_instance.gram_matrix(
        style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    content_weight = 1e4
    style_weight = 1e2

    optimizer = Adam([target], lr=0.003)

    for step in range(steps):
        target_features = model_instance.get_features(
            target, layers=[content_layer] + list(style_layers))

        content_loss = torch.mean(
            (target_features[content_layer] - target_features[content_layer]) ** 2)

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

    final_image = target.clone().cpu().detach()
    final_pil_image = Image.fromarray(
        (final_image.squeeze().permute(1, 2, 0).numpy() * 255).astype("uint8"))
    final_pil_image.save(output_image)
    print(f"Style transfer complete! image saved as '{output_image}'")
