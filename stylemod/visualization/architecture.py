import pkgutil
import importlib
import inspect
from stylemod.core.base import BaseModel
from stylemod.core.cnn import CNNBaseModel
from stylemod.core.transformer import TransformerBaseModel
from stylemod.visualization.gv import Graphviz, Style
from graphviz import Digraph


def visualize(show_funcs: bool = False, style=Style.MOLOKAI.value) -> Digraph:
    title = "stylemod"
    dg = Digraph(comment=title, graph_attr={"size": "3.25!"})

    style = Style.MOLOKAI.value
    Graphviz.stylize(dg, style=style)

    tr_font_size = style.custom.get(
        "tr_font_size", "8")
    subgraph_colors = style.custom.get("subgraph_colors", [])
    sg_color_1 = subgraph_colors[0] if len(subgraph_colors) > 0 else "darkgray"
    sg_color_2 = subgraph_colors[1] if len(subgraph_colors) > 1 else "gray"
    sg_font_color = style.custom.get(
        "sg_font_color", "white")
    semibold_font = style.custom.get(
        "semibold_font", style.font)
    title_font_size = style.custom.get(
        "title_font_size", "24")
    title_color = style.custom.get(
        "title_color", "purple")

    dg.node(
        "title",
        label=f'''<<font face="{style.font}" point-size="{title_font_size}" color="{sg_font_color}"><b>{title}</b></font>>''',
        shape="box",
        style="filled",
        color=title_color,
        fillcolor=style.node_fill,
        width="6.0" if show_funcs else "2.25",
        fixedsize="true",
    )

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
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">forward()</font></td>
            <td align="center" width="{td_width}"><font point-size="{tr_font_size}">visualize()</font></td>
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
    dg.edge("ABM", "BM", style="dashed")  # AbstractBaseModel -> BaseModel
    dg.edge("BM", "CBM")  # BaseModel -> CNNBaseModel
    dg.edge("BM", "TBM")  # BaseModel -> TransformerBaseModel

    return dg
