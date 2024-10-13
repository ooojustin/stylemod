import torch
import pkgutil
import importlib
import inspect
from stylemod.core.gv import Graphviz, Style
from stylemod.core.factory import ModelFactory
from stylemod.core.base import BaseModel
from stylemod.core.cnn import CNNBaseModel
from stylemod.core.transformer import TransformerBaseModel
from stylemod.models import Model
from stylemod import utils
from tqdm import tqdm
from typing import Union, Optional, Literal
from torch.optim.adam import Adam
from torch.optim.lbfgs import LBFGS
from PIL import Image
from graphviz import Digraph


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
    optimizer_type: Literal["adam", "lbfgs"] = "adam",
    return_type: Literal["tensor", "pil"] = "tensor",
    _print: bool = True
) -> Union[torch.Tensor, Image.Image]:
    if isinstance(model, Model):
        model = ModelFactory.create(model.name)
    elif isinstance(model, BaseModel):
        model = model
    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. Must be either a `Model` enum or a subclass of `BaseModel`.")

    device = utils.get_device(gpu_index, _print=_print)
    model.set_device(device)
    if model.eval_mode:
        model.eval()

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
        raise TypeError(
            f"Invalid type for content_image:  {type(content_image)}. Must be either a str or torch.Tensor.")

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
        raise TypeError(
            f"Invalid type for style_image:  {type(style_image)}. Must be either a str or torch.Tensor.")

    if model.normalization is not None:
        content = model.normalize_tensor(content)
        style = model.normalize_tensor(style)

    content_features = model.get_features(
        content, layers=[model.content_layer])
    style_features = model.get_features(style, layers=model.style_layers)

    if isinstance(model, TransformerBaseModel):
        model.compute_style_attention(style)

    target = content.clone().requires_grad_(True).to(device)

    if optimizer_type == "lbfgs":
        optimizer = LBFGS([target], max_iter=steps, lr=learning_rate)
    elif optimizer_type == "adam":
        optimizer = Adam([target], lr=learning_rate)

    def loss_step():
        total_loss = model.forward(
            target=target,
            content_features=content_features,
            style_features=style_features,
            content_weight=content_weight,
            style_weight=style_weight
        )
        total_loss.backward(retain_graph=model.retain_graph)
        return total_loss

    step_range = tqdm(
        range(steps), desc="Loss Optimization") if _print else range(steps)
    for step in step_range:
        if isinstance(optimizer, Adam):
            optimizer.zero_grad()
            total_loss = loss_step()
            optimizer.step()
        elif isinstance(optimizer, LBFGS):
            total_loss = optimizer.step(loss_step)
        else:
            raise AssertionError("Invalid optimizer.")

        if step % 10 == 0 and isinstance(step_range, tqdm):
            step_range.set_postfix(  # type: ignore
                {"total_loss": total_loss.item()})

    tensor = target.clone().cpu().detach()
    if return_type == "pil":
        if model.normalization is not None:
            tensor = model.denormalize_tensor(tensor)
        tensor = tensor.clamp(0, 1).squeeze().permute(1, 2, 0)
        arr = (tensor.numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr)
        return pil
    else:
        return tensor


def generate_class_hierarchy(show_funcs: bool = False) -> Digraph:
    title = "stylemod"
    dg = Digraph(comment=title, graph_attr={"size": "3.25!"})

    color_scheme = Style.MOLOKAI.value
    tr_font_size = color_scheme.custom.get(
        "tr_font_size", "8")
    sg_color_1 = color_scheme.custom.get(
        "soft_blue", "darkgray")
    sg_color_2 = color_scheme.custom.get(
        "slate_gray", "gray")
    sg_font_color = color_scheme.custom.get(
        "white", "white")
    semibold_font = color_scheme.custom.get(
        "semibold_font", color_scheme.font)
    title_font_size = color_scheme.custom.get(
        "title_font_size", "24")
    purple = color_scheme.custom.get(
        "purple", "purple")

    dg.node(
        "title",
        label=f'''<<font face="{color_scheme.font}" point-size="{title_font_size}" color="{sg_font_color}"><b>{title}</b></font>>''',
        shape="box",
        style="filled",
        color=purple,
        fillcolor=color_scheme.node_fill,
        width="6.0" if show_funcs else "2.25",
        fixedsize="true",
    )

    Graphviz.stylize(dg, style=color_scheme)

    # generic abstractions
    if show_funcs:
        td_width = 200
        dg.node("ABM", label=(
            f'''<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="8" width="{td_width * 2}">
            <tr>
            <td colspan="2" align="center" width="{td_width * 2}" style="dotted"><font face="{semibold_font}">AbstractBaseModel</font></td>
            </tr>
            <tr>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">initialize_module()</font></td>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">get_model_module()</font></td>
            </tr>
            <tr>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">eval()</font></td>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">set_device()</font></td>
            </tr>
            <tr>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">normalize_tensor()</font></td>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">denormalize_tensor()</font></td>
            </tr>
            <tr>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">get_features()</font></td>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">calc_gram_matrix()</font></td>
            </tr>
            <tr>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">calc_content_loss()</font></td>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">calc_style_loss()</font></td>
            </tr>
            <tr>
            <td align="center" colspan="2" width="{td_width * 2}"><font point-size="{tr_font_size}">forward()</font></td>
            </tr>
            </table>>'''
        ), shape="plaintext")
    else:
        dg.node(
            "ABM", label=f'''<<font face="{semibold_font}">AbstractBaseModel</font>>''')
    dg.node("BM", label=f'''<<font face="{semibold_font}">BaseModel</font>>''')

    # connect title to top node, invisible, for positioning
    dg.edge("title", "ABM", style="invis")

    # populate lists of cnn/transformer models for subgraphs
    cnn_models = []
    transformer_models = []
    pkg = "stylemod.models"
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(pkg).__path__):
        module = importlib.import_module(f"{pkg}.{module_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, '_noviz', False):
                continue
            if issubclass(obj, CNNBaseModel) and obj not in [BaseModel, CNNBaseModel]:
                cnn_models.append(name)
            elif issubclass(obj, TransformerBaseModel) and obj not in [BaseModel, TransformerBaseModel]:
                transformer_models.append(name)

    # subgraph for cnn based models
    with dg.subgraph(name="cluster_CNN") as cnn:  # type: ignore
        cnn.attr(label=f'''<<b>CNN Models</b>>''',
                 color=sg_color_1, fontcolor=sg_font_color)
        cnn.node(
            "CBM", label=f'''<<font face="{semibold_font}">CNNBaseModel</font>>''')

        for model_name in cnn_models:
            cnn.node(model_name, model_name)
            cnn.edge("CBM", model_name)

    # subgraph for transformer based models
    with dg.subgraph(name="cluster_Transformer") as transformer:  # type: ignore
        transformer.attr(label=f'''<<b>Transformer Models</b>>''',
                         color=sg_color_2, fontcolor=sg_font_color)

        if show_funcs:
            td_width = 30
            transformer.node("TBM", label=(
                f'''<
                <table border="0" cellborder="1" cellspacing="0" cellpadding="8">
                <tr><td><font face="{semibold_font}">TransformerBaseModel</font></td></tr>
                <tr><td align="center"><font point-size="{tr_font_size}">get_attention()</font></td></tr>
                <tr><td align="center"><font point-size="{tr_font_size}">compute_style_attention()</font></td></tr>
                </table>>'''
            ), shape="plaintext")
        else:
            transformer.node(
                "TBM", label=f'''<<font face="{semibold_font}">TransformerBaseModel</font>>''')

        for model_name in transformer_models:
            transformer.node(model_name, model_name)
            transformer.edge("TBM", model_name)

    # connect high level nodes
    dg.edge("ABM", "BM")  # AbstractBaseModel -> BaseModel
    dg.edge("BM", "CBM")  # BaseModel -> CNNBaseModel
    dg.edge("BM", "TBM")  # BaseModel -> TransformerBaseModel

    return dg
