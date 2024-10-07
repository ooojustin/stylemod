import torch
from PIL import Image
import click
from torchvision import transforms, models

vgg = models.vgg19(pretrained=True).features


def load_image(img_path, max_size=400):
    image = Image.open(img_path)
    size = max_size if max(image.size) > max_size else max(image.size)
    image = image.resize((size, int(size * image.size[1] / image.size[0])))
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image


def get_features(image, model, layers=None):
    # NOTE(justin): key 21 is content layer
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


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


@click.command()
@click.option('--content-image', required=True, help='Path to the content image.')
@click.option('--style-image', required=True, help='Path to the style image.')
def style_transfer(content_image, style_image):
    content = load_image(content_image)
    style = load_image(style_image)

    # extract features for both content and style
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate gram matrix for style features
    style_grams = {layer: gram_matrix(
        style_features[layer]) for layer in style_features}

    print(
        f"Extracted content features and calculated style Gram matrix for: {style_image}")
