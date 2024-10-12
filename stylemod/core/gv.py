import subprocess
import platform
from enum import Enum
from graphviz import Digraph
from typing import NamedTuple, Dict


class ColorSchemeType(NamedTuple):
    background: str
    node_fill: str
    node_border: str
    node_font: str
    edge_color: str
    font: str
    custom: Dict[str, str] = {}


class ColorScheme(Enum):
    MOLOKAI = ColorSchemeType(
        background="#1B1D1E",
        node_fill="#272822",
        node_border="#66D9EF",
        node_font="#F8F8F2",
        edge_color="#A6E22E",
        font="Tahoma",
        custom={
            "muted_cyan": "#5588AA",
            "soft_blue": "#3A3D43",
            "slate_gray": "#708090"
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
                    "Please manually install Graphviz and ensure the 'dot' executable is available in your system's PATH."
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
            subprocess.run(['brew', '--version'], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['brew', 'install', 'graphviz'], check=True)
            print("Graphviz installed successfully via Homebrew.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Homebrew is not installed. Please install Homebrew and try again.")
            print("You can install Homebrew from https://brew.sh/")

    @staticmethod
    def install_linux():
        """Install Graphviz on Linux using apt-get or yum depending on the distribution."""
        try:
            # try apt-get (debian/ubuntu)
            subprocess.run(['sudo', 'apt-get', 'install',
                           '-y', 'graphviz'], check=True)
            print("Graphviz installed successfully via apt-get.")
        except subprocess.CalledProcessError:
            try:
                # try yum (red hat)
                subprocess.run(["sudo", "yum", "install",
                               "-y", "graphviz"], check=True)
                print("Graphviz installed successfully via yum.")
            except subprocess.CalledProcessError:
                print(
                    "Please manually install Graphviz using your system's package manager.")

    @staticmethod
    def stylize(dg: Digraph, colors: ColorSchemeType = ColorScheme.MOLOKAI.value):
        dg.attr(rankdir="TB",
                size="10",
                fontname=colors.font,
                fontsize="12",
                style="filled",
                bgcolor=colors.background,
                color=colors.node_font)

        dg.attr("node",
                shape="box",
                style="filled",
                fillcolor=colors.node_fill,
                fontname=colors.font,
                fontsize="10",
                color=colors.node_border,
                fontcolor=colors.node_font)

        dg.attr("edge",
                color=colors.edge_color,
                style="solid",
                arrowhead="open",
                fontname=colors.font,
                fontsize="10")
