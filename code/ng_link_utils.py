"""
ng_link_utils.py — Neuroglancer link generation for molmo-glancer

Builds neuroglancer-demo.appspot.com links from zarr S3 paths using the
neuroglancer Python package for proper state validation.

Primary entry points:
    make_zarr_link(zarr_s3_path)          → single grayscale layer link
    make_multichannel_link(channels)       → multiple colored layers link

CLI:
    python /code/ng_link_utils.py
    python /code/ng_link_utils.py --zarr s3://bucket/path/data.zarr
    python /code/ng_link_utils.py --zarr s3://... --layout xy --save
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import neuroglancer

# ── defaults ─────────────────────────────────────────────────────────────────

NG_HOST = "https://neuroglancer-demo.appspot.com/"

# Default zarr: example AIND mFISH dataset (from example_ng_link.txt)
DEFAULT_ZARR = (
    "s3://aind-open-data/HCR_772643-3a-1_2025-03-19_10-00-00"
    "/SPIM.ome.zarr/Tile_X_0000_Y_0000_Z_0000_ch_405.zarr/"
)

# Hardcoded LUT range for this dataset
DEFAULT_LUT = [0, 400]

EXAMPLE_LINK_PATH = Path("/root/capsule/example_ng_link.txt")


# ── dimension configs ─────────────────────────────────────────────────────────

def _dims_xyzt() -> neuroglancer.CoordinateSpace:
    """x/y/z at 1 µm, t at 1 ms — matches the AIND SPIM/mFISH dataset."""
    return neuroglancer.CoordinateSpace({
        "x": [1e-6, "m"],
        "y": [1e-6, "m"],
        "z": [1e-6, "m"],
        "t": [0.001, "s"],
    })


# ── shader helpers ────────────────────────────────────────────────────────────

def _grayscale_shader() -> str:
    return "#uicontrol invlerp normalized\nvoid main(){emitGrayscale(normalized());}"


def _color_shader(hex_color: str) -> str:
    """Colored shader: intensity mapped to a single hue."""
    return (
        f'#uicontrol vec3 color color(default="{hex_color}")\n'
        "#uicontrol invlerp normalized\n"
        "void main(){emitRGB(color * normalized());}"
    )


def _shader_controls(lut: List[int]) -> dict:
    return {"normalized": {"range": lut}}


# ── layer builders ────────────────────────────────────────────────────────────

def _image_layer(
    zarr_s3_path: str,
    name: str,
    lut: List[int],
    color: Optional[str] = None,
    visible: bool = True,
    opacity: float = 1.0,
    blend: str = "default",
) -> dict:
    """Build a validated neuroglancer ImageLayer dict.

    Parameters
    ----------
    zarr_s3_path:
        S3 path, with or without 's3://' prefix. The 'zarr://' scheme
        prefix is added automatically.
    name:
        Display name in the neuroglancer layer panel.
    lut:
        [min, max] intensity range for the invlerp control.
    color:
        Hex color string (e.g. '#FF00FF') for a colored shader.
        If None, uses grayscale.
    visible:
        Whether the layer is visible by default.
    """
    path = zarr_s3_path if zarr_s3_path.startswith("s3://") else f"s3://{zarr_s3_path}"
    source = f"zarr://{path}"

    cfg = {
        "name": name,
        "source": source,
        "shader": _color_shader(color) if color else _grayscale_shader(),
        "shaderControls": _shader_controls(lut),
        "visible": visible,
    }
    if opacity != 1.0:
        cfg["opacity"] = opacity
    if blend != "default":
        cfg["blend"] = blend

    cfg["type"] = "image"
    return cfg


# ── link builders ─────────────────────────────────────────────────────────────

def make_zarr_link(
    zarr_s3_path: str = DEFAULT_ZARR,
    name: Optional[str] = None,
    lut: List[int] = DEFAULT_LUT,
    color: Optional[str] = None,
    layout: str = "4panel",
) -> str:
    """Build a neuroglancer link for a single zarr source.

    Parameters
    ----------
    zarr_s3_path:
        S3 path to the zarr array.
    name:
        Layer name shown in the UI. Defaults to the last path component.
    lut:
        [min, max] intensity clamp. Default: [200, 800].
    color:
        Optional hex color for a colored shader (e.g. '#00FFFF').
        None → grayscale.
    layout:
        Neuroglancer layout. '4panel' (default), 'xy', 'xz', 'yz', '3d'.

    Returns
    -------
    str
        Full neuroglancer URL ready to open in a browser.
    """
    if name is None:
        # use last non-empty path component as the layer name
        name = next(
            (p for p in reversed(zarr_s3_path.rstrip("/").split("/")) if p),
            "layer",
        )

    layer = _image_layer(zarr_s3_path, name=name, lut=lut, color=color)

    state = {
        "dimensions": _dims_xyzt().to_json(),
        "layers": [layer],
        "layout": layout,
    }

    return NG_HOST + "#!" + json.dumps(state)


def make_multichannel_link(
    channels: List[Tuple[str, str, str]],
    lut: List[int] = DEFAULT_LUT,
    layout: str = "4panel",
) -> str:
    """Build a neuroglancer link with multiple colored image layers.

    Parameters
    ----------
    channels:
        List of (zarr_s3_path, layer_name, hex_color) tuples.
        Example:
            [
                ("s3://bucket/ch_405.zarr", "DAPI",   "#0000FF"),
                ("s3://bucket/ch_488.zarr", "GFP",    "#00FF00"),
                ("s3://bucket/ch_561.zarr", "mCherry","#FF0000"),
            ]
    lut:
        Shared [min, max] intensity range applied to all channels.
    layout:
        Neuroglancer layout string.

    Returns
    -------
    str
        Full neuroglancer URL.
    """
    layers = [
        _image_layer(path, name=name, lut=lut, color=color)
        for path, name, color in channels
    ]

    state = {
        "dimensions": _dims_xyzt().to_json(),
        "layers": layers,
        "layout": layout,
    }

    return NG_HOST + "#!" + json.dumps(state)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Neuroglancer link for a zarr S3 path."
    )
    parser.add_argument(
        "--zarr",
        default=DEFAULT_ZARR,
        help=f"S3 path to zarr array (default: {DEFAULT_ZARR})",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Layer name in the UI (default: last path component)",
    )
    parser.add_argument(
        "--lut",
        nargs=2,
        type=int,
        default=DEFAULT_LUT,
        metavar=("MIN", "MAX"),
        help=f"Intensity range, e.g. --lut 200 800 (default: {DEFAULT_LUT})",
    )
    parser.add_argument(
        "--color",
        default=None,
        metavar="HEX",
        help="Hex color for colored shader, e.g. #00FFFF (default: grayscale)",
    )
    parser.add_argument(
        "--layout",
        default="4panel",
        choices=["4panel", "xy", "xz", "yz", "3d"],
        help="Viewer layout (default: 4panel)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Overwrite {EXAMPLE_LINK_PATH} with the generated link",
    )
    args = parser.parse_args()

    link = make_zarr_link(
        zarr_s3_path=args.zarr,
        name=args.name,
        lut=args.lut,
        color=args.color,
        layout=args.layout,
    )

    print(link)

    if args.save:
        EXAMPLE_LINK_PATH.write_text(link + "\n")
        print(f"\nSaved to {EXAMPLE_LINK_PATH}")


if __name__ == "__main__":
    main()
