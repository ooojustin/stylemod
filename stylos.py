import torch
from PIL import Image
import click
from torchvision import transforms


def load_image(img_path, max_size=400):
    image = Image.open(img_path)
    size = max_size if max(image.size) > max_size else max(image.size)
    image = image.resize((size, int(size * image.size[1] / image.size[0])))
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image


@click.command()
@click.option('--content-image', required=True, help='Path to the content image.')
@click.option('--style-image', required=True, help='Path to the style image.')
def style_transfer(content_image, style_image):
    content = load_image(content_image)
    style = load_image(style_image)
    print(f"Loaded content image: {content_image}, style image: {style_image}")


if __name__ == '__main__':
    style_transfer()
