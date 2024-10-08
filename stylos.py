import torch
from torch.optim.adam import Adam
from PIL import Image
import click
from torchvision import transforms, models
from torchvision.models import VGG19_Weights


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


# load the pre-trained vgg19 model
vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
for param in vgg.parameters():
    param.requires_grad_(False)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg.to(device)


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


# extract features from vgg19
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


# calculate the gram matrix for style loss
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


@click.command()
@click.option("--content-image", required=True, help="Path to the content image.")
@click.option("--style-image", required=True, help="Path to the style image.")
@click.option("--output-image", default="output_image.png", help="filename for the output image.")
@click.option("--steps", default=2000, help="Number of optimization steps (default: 2000).")
@click.option("--max-size", default=400, help="Maximum size of input images (default: 400).")
@click.option("--gpu-index", default=None, type=int, help="GPU index to use (default: 0 if available).")
def style_transfer(content_image, style_image, output_image, steps, max_size, gpu_index):
    list_available_gpus()
    device = get_device(gpu_index)

    content = load_image(content_image, max_size=max_size).to(device)
    style = load_image(
        style_image, shape=content.shape[-2:], max_size=max_size).to(device)

    # extract features for both content and style
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate gram matrix for style features
    style_grams = {layer: gram_matrix(
        style_features[layer]) for layer in style_features}

    # create target image (copy of content) for optimization
    target = content.clone().requires_grad_(True).to(device)

    # define weights for content and style loss
    style_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.8,
        "conv3_1": 0.5,
        "conv4_1": 0.3,
        "conv5_1": 0.1
    }

    content_weight = 1e4
    style_weight = 1e2

    optimizer = Adam([target], lr=0.003)

    # optimization loop
    for step in range(steps):
        target_features = get_features(target, vgg)

        # content loss
        content_loss = torch.mean(
            (target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

        # style loss
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * \
                torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / \
                (target_feature.size(1) * target_feature.size(2))

        total_loss = content_weight * content_loss + style_weight * style_loss

        # update the target image
        optimizer.zero_grad()
        total_loss.backward()
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


if __name__ == "__main__":
    style_transfer()
