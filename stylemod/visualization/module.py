import torch
import graphviz
from stylemod.core.base import BaseModel
from stylemod.visualization.gv import Graphviz, Style


def visualize(model: BaseModel, style=Style.MOLOKAI.value) -> graphviz.Digraph:
    model.get_model_module()
    assert isinstance(model.model, torch.nn.Module)

    dot = graphviz.Digraph(
        comment=f"Model: {model.__class__.__name__}", format="png")
    Graphviz.stylize(dot, style=style)

    layer_kwargs = style.custom.get("layer_kwargs", {})
    conv2d_kwargs = layer_kwargs.get("conv2d", {})
    relu_kwargs = layer_kwargs.get("relu", {})
    maxpool2d_kwargs = layer_kwargs.get("maxpool2d", {})
    linear_kwargs = layer_kwargs.get("linear", {})
    other_kwargs = layer_kwargs.get("other", {})

    def add_layer_to_graph(module: torch.nn.Module, parent_name: str = "", block_num: int = 0):
        conv_count = 0
        prev_layer_name = None
        for name, layer in module.named_children():
            layer_name = f"{parent_name}/{name}" if parent_name else name
            label = f"{name}: {layer.__class__.__name__}"
            if isinstance(layer, torch.nn.Conv2d):
                label += f" (Conv Block {block_num}, Conv {conv_count})"
                dot.node(layer_name, label, **conv2d_kwargs)
                conv_count += 1
            elif isinstance(layer, torch.nn.ReLU):
                dot.node(layer_name, label, **relu_kwargs)
            elif isinstance(layer, torch.nn.MaxPool2d):
                dot.node(layer_name, label, **maxpool2d_kwargs)
                block_num += 1
                conv_count = 0
            elif isinstance(layer, torch.nn.Linear):
                dot.node(layer_name, label, **linear_kwargs)
            else:
                # fallback
                dot.node(layer_name, label, **other_kwargs)

            if prev_layer_name:
                dot.edge(prev_layer_name, layer_name)

            if len(list(layer.named_children())) > 0:
                add_layer_to_graph(layer, layer_name, block_num)

            prev_layer_name = layer_name

    add_layer_to_graph(model.model)

    return dot
