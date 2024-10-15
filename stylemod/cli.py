import os
import io
import click
import stylemod
from stylemod.core.factory import ModelFactory
from stylemod.visualization import architecture
from stylemod.visualization.gv import Graphviz
from stylemod.models import Model
from typing import Optional
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
def run(
    content_image: str,
    style_image: str,
    output_image: str,
    steps: int,
    max_size: int,
    model: str,
    gpu_index: Optional[int]
) -> None:
    model_enum = Model[model]
    print("Model:", model_enum.name)
    output = stylemod.style_transfer(
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


@click.command()
@click.option("--save", "-s", is_flag=True, help="Save the rendered class hierarchy to a file.")
@click.option("--show-funcs", "-f", is_flag=True, help="Show abstract functions that should be implemented by this nodes subclasses.")
@click.option("--dpi", "-d", default=200, help="Set the DPI (dots per inch) for the rendered image. [Default: 200]")
def class_hierarchy(save: bool, show_funcs: bool, dpi: int):
    Graphviz.install()

    img_dir = "img"
    if save and not os.path.exists(img_dir):
        os.makedirs(img_dir)

    dot = architecture.visualize(show_funcs=show_funcs)
    dot.attr(dpi=str(dpi))
    png = dot.pipe(format="png")

    if save:
        png_path = os.path.join(img_dir, "class_hierarchy.png")
        dot_path = "stylemod.dot"
        with open(png_path, "wb") as f:
            f.write(png)
        with open("stylemod.dot", "w") as f:
            f.write(dot.source)
        click.echo(
            f"Class hierarchy visualization saved as '{png_path}', dot file saved as '{dot_path}'.")

    image = Image.open(io.BytesIO(png))
    image.show()


@click.command()
@click.argument("model", type=CaseInsensitiveChoice([model.name for model in Model]))
@click.option("--output", default=None, help="Optional path to save the visualization image (e.g., 'model_vis.png'). If not provided, it will just display.")
@click.option("--dpi", "-d", default=400, help="Set the DPI (dots per inch) for the rendered image. [Default: 400]")
def visualize(model: str, output: Optional[str], dpi: int) -> None:
    """
    Visualizes the architecture of a given model.
    If the output is provided, saves the visualization as an image file.
    Otherwise, it displays the model graph.
    """
    model_enum = Model[model]
    model_instance = ModelFactory.create(model_enum.name)

    dot = model_instance.visualize()
    dot.attr(dpi=str(dpi))

    if output:
        output_format = output.split(".")[-1]
        dot.render(output, format=output_format, cleanup=True)
        click.echo(f"Model visualization saved to '{output}'.")
    else:
        png_data = dot.pipe(format="png")
        image = Image.open(io.BytesIO(png_data))
        image.show()


@click.group()
def cli():
    pass


cli.add_command(run)
cli.add_command(class_hierarchy)
cli.add_command(visualize)
