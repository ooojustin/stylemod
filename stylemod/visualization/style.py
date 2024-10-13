from enum import Enum
from typing import NamedTuple, Dict


class StyleType(NamedTuple):
    node_fill: str
    node_border: str
    node_font: str
    edge_color: str
    font: str
    font_sz_graph: str = "12"
    font_sz_node: str = "10"
    font_sz_edge: str = "10"
    background: str = "transparent"
    rankdir: str = "TB"
    splines: str = "true"
    custom: Dict[str, str] = {}


class Style(Enum):
    MOLOKAI = StyleType(
        # background="#1B1D1E",
        node_fill="#272822",
        node_border="#66D9EF",
        node_font="#F8F8F2",
        edge_color="#A6E22E",
        font="Fira Code, Segoe UI, Helvetica, Arial",
        font_sz_graph="16",
        font_sz_node="12",
        font_sz_edge="12",
        splines="ortho",
        custom={
            "semibold_font": "Fira Code Medium, Segoe UI Semibold, Helvetica Neue Medium, Arial Semibold",
            "title_font_size": "20",
            "tr_font_size": "10",
            "purple": "#AE81FF",
            "soft_blue": "#3A3D43",
            "slate_gray": "#708090",
            "white": "#F0F0F0"
        }
    )
