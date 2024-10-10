import click
import torch
from stylemod.models import Models
from torchvision import transforms
from torch.optim.adam import Adam
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


@click.command()
@click.option("--content-image", required=True, help="Path to the content image.")
@click.option("--style-image", required=True, help="Path to the style image.")
@click.option("--output-image", default="output_image.png", help="Filename for the output image. [Default: output_image.png]")
@click.option("--steps", default=2000, help="Number of optimization steps. [Default: 2000]")
@click.option("--max-size", default=400, help="Maximum size of input images. [Default: 400]")
@click.option("--model", default="vgg19", type=str, help="Model to use for feature extraction. [Default: vgg19]")
@click.option("--gpu-index", default=None, type=int, help="GPU index to use. [Default: 0, if available]")
def style_transfer(content_image, style_image, output_image, steps, max_size, model, gpu_index):
    list_available_gpus()
    device = get_device(gpu_index)

    model = Models.load(model)
    model.set_device(device)
    if model.eval_mode:
        model.eval()

    # model_module = model.get_model_module()
    # print(model_module)
    print("Model:", model.name)

    content_layer = model.content_layer
    style_layers = model.style_layers
    style_weights = model.style_weights

    content = load_image(content_image, max_size=max_size).to(device)
    style = load_image(
        style_image, shape=content.shape[-2:], max_size=max_size).to(device)

    # extract features for both content and style
    content_features = model.get_features(content, layers=[content_layer])
    style_features = model.get_features(style, layers=style_layers)

    # content loss using the models designated content layer
    content_loss = torch.mean(
        (content_features[content_layer] - content_features[content_layer]) ** 2)

    # calculate gram matrix for style features
    style_grams = {layer: model.gram_matrix(
        style_features[layer]) for layer in style_features}

    # create target image (copy of content) for optimization
    target = content.clone().requires_grad_(True).to(device)

    content_weight = 1e4
    style_weight = 1e2

    optimizer = Adam([target], lr=0.003)

    # optimization loop
    for step in range(steps):
        target_features = model.get_features(
            target, layers=[content_layer] + style_layers)

        # calculate content loss
        content_loss = torch.mean(
            (target_features[content_layer] - content_features[content_layer]) ** 2)

        # calculate style loss
        style_loss = 0
        for layer in style_layers:
            target_gram = model.gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += style_weights[layer] * \
                torch.mean((target_gram - style_gram) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        # update the target image
        optimizer.zero_grad()
        total_loss.backward(retain_graph=model.retain_graph)
        optimizer.step()

        # log the loss every 10 steps
        if step % 10 == 0:
            print(f"Step {step}, total loss: {total_loss.item()}")

    # save the final output image
    final_image = target.clone().cpu().detach()
    final_pil_image = Image.fromarray(
        (final_image.squeeze().permute(1, 2, 0).numpy() * 255).astype("uint8"))
    final_pil_image.save(output_image)
    print(f"Style transfer complete! image saved as '{output_image}'")
