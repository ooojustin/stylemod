import click
from stylemod import style_transfer
from stylemod.models import Model
from PIL import Image


class CaseInsensitiveChoice(click.Choice):
    def convert(self, value, param, ctx):
        value = value.upper()
        if value in [choice.upper() for choice in self.choices]:
            return value
        self.fail(
            f"Invalid choice: {value}. (choose from {', '.join(self.choices)})", param, ctx)


@click.command()
@click.option("--content-image", required=True, help="Path to the content image.")
@click.option("--style-image", required=True, help="Path to the style image.")
@click.option("--output-image", default="output_image.png", help="Filename for the output image. [Default: output_image.png]")
@click.option("--steps", default=1000, help="Number of optimization steps. [Default: 1000]")
@click.option("--max-size", default=400, help="Maximum size of input images. [Default: 400]")
@click.option("--model", type=CaseInsensitiveChoice([model.name for model in Model]), default="VGG19", help="Model to use for feature extraction. [Default: VGG19]")
@click.option("--gpu-index", default=None, type=int, help="GPU index to use. [Default: 0, if available]")
def run(content_image, style_image, output_image, steps, max_size, model, gpu_index):
    model_enum = Model[model]
    print("Model:", model_enum.name)
    output = style_transfer(
        content_image=content_image,
        style_image=style_image,
        steps=steps,
        max_size=max_size,
        model=model_enum,
        gpu_index=gpu_index,
        return_type="pil"
    )
    assert isinstance(output, Image.Image)
    output.save(output_image)
