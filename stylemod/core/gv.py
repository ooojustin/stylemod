import subprocess
import platform
from enum import Enum
from graphviz import Digraph
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


class Graphviz:

    @staticmethod
    def install():
        try:
            subprocess.run(["dot", "-V"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Graphviz not found. Installing...")
            system = platform.system().lower()
            if system == "windows":
                Graphviz.install_win()
            elif system == "darwin":
                Graphviz.install_macos()
            elif system == "linux":
                Graphviz.install_linux()
            else:
                raise OSError(
                    f"Unsupported operating system detected: {system}.\n"
                    "Please manually install Graphviz and ensure the dot executable is available in your systems PATH."
                )

    @staticmethod
    def install_win():
        url = "https://graphviz.gitlab.io/_pages/Download/Download_windows.html"
        print(f"Please download and install Graphviz from {url}")
        print(
            "Once installed, make sure to add Graphviz to your PATH environment variable.")
        input("Press Enter after installing Graphviz to continue...")

    @staticmethod
    def install_macos():
        try:
            subprocess.run(["brew", "--version"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["brew", "install", "graphviz"], check=True)
            print("Graphviz installed successfully via Homebrew.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Homebrew is not installed. Please install Homebrew and try again.")
            print("You can install Homebrew from https://brew.sh/")

    @staticmethod
    def install_linux():
        """Install Graphviz on Linux using apt-get or yum depending on the distribution."""
        try:
            # try apt-get (debian/ubuntu)
            subprocess.run(["sudo", "apt-get", "install",
                           "-y", "graphviz"], check=True)
            print("Graphviz installed successfully via apt-get.")
        except subprocess.CalledProcessError:
            try:
                # try yum (red hat)
                subprocess.run(["sudo", "yum", "install",
                               "-y", "graphviz"], check=True)
                print("Graphviz installed successfully via yum.")
            except subprocess.CalledProcessError:
                print(
                    "Please manually install Graphviz using your systems package manager.")

    @staticmethod
    def stylize(dg: Digraph, style: StyleType = Style.MOLOKAI.value):
        # graph attributes
        dg.attr(rankdir=style.rankdir,
                size="10",
                fontname=style.font,
                fontsize=style.font_sz_graph,
                style="filled",
                bgcolor=style.background,
                color=style.node_font,
                splines=style.splines)

        # node attributes
        dg.attr("node",
                shape="box",
                style="filled",
                fillcolor=style.node_fill,
                fontname=style.font,
                fontsize=style.font_sz_node,
                color=style.node_border,
                fontcolor=style.node_font)

        # edge attributes
        dg.attr("edge",
                color=style.edge_color,
                style="solid",
                arrowhead="open",
                fontname=style.font,
                fontsize=style.font_sz_edge)


def noviz(cls):
    """Decorator to flag classes that should be excluded from graph visualization."""
    cls._noviz = True
    return cls
