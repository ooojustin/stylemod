import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from stylemod.visualization.style import Style, StyleType
from typing import List


def plot_losses(
    content_losses: List[float],
    style_losses: List[float],
    total_losses: List[float],
    style: StyleType = Style.MOLOKAI.value
):
    fonts = [font.strip() for font in style.font.split(",")]
    font_candidates = fonts + ["DejaVu Sans", "Arial", "sans-serif"]

    font_name = None
    for font in font_candidates:
        if font in fm.findSystemFonts(fontext='ttf'):
            font_name = font
            break

    if font_name is None:
        font_name = "DejaVu Sans"  # fallback

    plt.figure(figsize=(10, 5))

    plt.gcf().set_facecolor(style.background if style.background !=
                            "transparent" else "#000000")

    plt.plot(content_losses, label="Content Loss", color=style.node_font)
    plt.plot(style_losses, label="Style Loss", color=style.edge_color)
    plt.plot(total_losses, label="Total Loss", color=style.node_border)

    plt.title("Loss Progression",
              color=style.custom["title_color"], fontname=font_name, fontsize=style.custom["title_font_size"])

    plt.xlabel("Steps", fontsize=style.font_sz_graph,
               fontname=font_name, color=style.node_font)
    plt.ylabel("Loss", fontsize=style.font_sz_graph,
               fontname=font_name, color=style.node_font)

    ax = plt.gca()
    ax.set_facecolor(style.node_fill)
    ax.spines["bottom"].set_color(style.node_border)
    ax.spines["top"].set_color(style.node_border)
    ax.spines["left"].set_color(style.node_border)
    ax.spines["right"].set_color(style.node_border)
    ax.tick_params(axis="x", colors=style.node_font)
    ax.tick_params(axis="y", colors=style.node_font)

    legend = plt.legend(facecolor=style.node_fill,
                        edgecolor=style.node_border, fontsize=style.font_sz_node)
    for text in legend.get_texts():
        text.set_color(style.node_font)

    plt.grid(True, color=style.edge_color, linestyle='--', linewidth=0.5)
    plt.show()
