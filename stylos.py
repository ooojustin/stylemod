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
    if layers is None:
        layers = {'21': 'conv4_2'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


@click.command()
@click.option('--content-image', required=True, help='Path to the content image.')
@click.option('--style-image', required=True, help='Path to the style image.')
def style_transfer(content_image, style_image):
    content = load_image(content_image)
    content_features = get_features(content, vgg)
    print(f"Extracted content features from: {content_image}")


if __name__ == '__main__':
    style_transfer()
