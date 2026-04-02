"""
volume_info — Volume metadata discovery from Neuroglancer state + zarr sources.

Reads shape, voxel scales, layer info from the NG state and zarr .zarray metadata.
Provides FOV computation for scale-aware prompting.
"""

import json
from dataclasses import dataclass, field

VIEWPORT_SIZE = 1024

# Named zoom levels — the model picks one of these instead of a raw float.
# Defined as multipliers of fit_scale (which fills the full data extent).
ZOOM_LEVELS = {
    "wide":         2.0,    # zoomed out, all data visible with margin
    "full":         1.0,    # data fills the screen
    "region":       0.5,    # CROPS: ~half the data visible, edges excluded
    "close-up":     0.25,   # CROPS: ~quarter visible, most data excluded
    "single-cell":  0.125,  # CROPS: tiny fraction visible, nearly all data excluded
}


def build_zoom_table(volume_info: "VolumeInfo") -> dict[str, dict]:
    """Build a zoom table with concrete values for a specific volume."""
    fit_scale = max(volume_info.shape[0], volume_info.shape[1]) / VIEWPORT_SIZE
    table = {}
    for name, multiplier in ZOOM_LEVELS.items():
        scale = fit_scale * multiplier
        fov_um = scale * VIEWPORT_SIZE
        table[name] = {
            "crossSectionScale": scale,
            "fov_um": fov_um,
            "multiplier": multiplier,
        }
    return table


def resolve_zoom(zoom_name: str, volume_info: "VolumeInfo") -> float:
    """Convert a zoom level name to a crossSectionScale value."""
    fit_scale = max(volume_info.shape[0], volume_info.shape[1]) / VIEWPORT_SIZE
    multiplier = ZOOM_LEVELS.get(zoom_name)
    if multiplier is not None:
        return fit_scale * multiplier
    # Fallback: try parsing as float
    try:
        return float(zoom_name)
    except (ValueError, TypeError):
        return fit_scale  # default to fit


def format_zoom_table() -> str:
    """Format the zoom options for inclusion in the prompt."""
    return """ZOOM OPTIONS (use one of these names for "zoom"):
  "wide"        — zoomed out, entire volume visible with margin (~2× FOV)
  "full"        — entire volume fills the screen; use this to survey the full population
  --- below "full", the view CROPS the data — you will NOT see everything ---
  "region"      — ⚠ CROPS to ~half the volume; edges are excluded from view
  "close-up"    — ⚠ CROPS to ~quarter; most of the data is NOT visible
  "single-cell" — ⚠ CROPS to a tiny area; nearly all data excluded, for fine detail only"""


@dataclass
class LayerInfo:
    """Metadata about a single NG layer."""
    name: str
    type: str                           # "image", "segmentation", "annotation"
    source: str                         # zarr source URL
    extent: list[float] | None         # [X, Y, Z] physical extent, or None if unknown
    visible: bool
    shader_range: list[float] | None   # [vmin, vmax] if available


@dataclass
class VolumeInfo:
    """Metadata about the volume being analyzed."""
    bounding_box: list[float]           # [X, Y, Z] extent of visible layers
    voxel_scales: list[float]           # [sx, sy, sz] in meters
    axis_names: list[str]               # ["x", "y", "z"] or ["x", "y", "z", "t"]
    layers: list[LayerInfo]             # per-layer metadata
    canonical_factors: list[float]      # [fx, fy, fz] — scale / min(scale)
    anisotropy_ratio: float             # max(factors) / min(factors)

    @property
    def shape(self) -> list[float]:
        """Extent based on visible layers (backward compatible)."""
        visible = [l for l in self.layers if l.visible and l.extent]
        if visible:
            return [
                max(l.extent[i] for l in visible)
                for i in range(3)
            ]
        return self.bounding_box

    def format_for_prompt(self) -> str:
        """Format volume info for inclusion in the agent decision prompt."""
        s = self.shape
        cx, cy, cz = s[0] / 2, s[1] / 2, s[2] / 2
        units = "\u00b5m"

        lines = [
            f"Size: {s[0]:.0f} \u00d7 {s[1]:.0f} \u00d7 {s[2]:.0f} {units} (visible layers)",
        ]
        if self.anisotropy_ratio > 1.5:
            lines.append(f"Note: z is {self.anisotropy_ratio:.1f}\u00d7 coarser than x/y")

        # Per-layer listing
        lines.append("Layers:")
        for l in self.layers:
            vis = "visible" if l.visible else "hidden"
            if l.extent:
                ext = f"{l.extent[0]:.0f}\u00d7{l.extent[1]:.0f}\u00d7{l.extent[2]:.0f} {units}"
            else:
                ext = "extent unknown"
            lines.append(f"  - {l.name} ({l.type}, {ext}) [{vis}]")

        lines.append(f"Center: x={cx:.1f}, y={cy:.1f}, z={cz:.1f}")
        lines.append(f"Ranges: x=[0..{s[0]:.0f}], y=[0..{s[1]:.0f}], z=[0..{s[2]:.0f}]")

        # Pixel size context at full zoom
        um_per_pixel = max(s[0], s[1]) / VIEWPORT_SIZE
        neuron_um = 30.0
        neuron_pixels = neuron_um / um_per_pixel
        lines.append(
            f"A neuron is ~{neuron_um:.0f}{units} across "
            f"(~{neuron_pixels:.0f}px at full zoom). "
            f"{'Objects this small need zoomed-in views for reliable detection.' if neuron_pixels < 40 else 'Visible at full zoom.'}"
        )
        return "\n  ".join(lines)


def _get_layer_source(layer: dict) -> str:
    """Extract the source URL string from an NG layer dict."""
    source = layer.get("source", "")
    if isinstance(source, dict):
        return source.get("url", "")
    return source


def _get_layer_visibility(layer: dict) -> bool:
    """Check if a layer is visible (default True if not specified)."""
    # NG uses "visible" key; absent means visible
    return layer.get("visible", True)


def _get_shader_range(layer: dict) -> list[float] | None:
    """Extract shader range from a layer's shaderControls if present."""
    sc = layer.get("shaderControls", {})
    normalized = sc.get("normalized", {})
    r = normalized.get("range")
    if isinstance(r, list) and len(r) == 2:
        return [float(r[0]), float(r[1])]
    return None


def discover_volume(ng_state: dict) -> VolumeInfo:
    """Extract volume metadata from an NG state dict.

    Reads dimensions and per-layer info. Attempts to read shape from each
    layer's zarr source. Computes bounding box from all layers.
    """
    dims = ng_state.get("dimensions", {})
    axis_names = list(dims.keys())
    voxel_scales = [v[0] for v in dims.values()]

    # ── Discover per-layer metadata ────────────────────────────────────
    layers = []
    for layer in ng_state.get("layers", []):
        source = _get_layer_source(layer)
        visible = _get_layer_visibility(layer)
        shader_range = _get_shader_range(layer)

        extent = None
        if isinstance(source, str) and "zarr" in source:
            extent = read_shape_from_source(source)

        layers.append(LayerInfo(
            name=layer.get("name", "unknown"),
            type=layer.get("type", "image"),
            source=source,
            extent=extent,
            visible=visible,
            shader_range=shader_range,
        ))

    # ── Compute bounding box (union of all layers with known extent) ──
    layers_with_extent = [l for l in layers if l.extent]
    if layers_with_extent:
        bounding_box = [
            max(l.extent[i] for l in layers_with_extent)
            for i in range(3)
        ]
    else:
        # Fallback: infer from position
        print("  WARNING: No layer shapes discovered. Using position-based estimate.")
        pos = ng_state.get("position", [0, 0, 0])
        bounding_box = [max(int(p * 2), 1000) for p in pos[:3]]

    # ── Compute canonical factors (for FOV computation) ────────────────
    spatial_scales = voxel_scales[:3] if len(voxel_scales) >= 3 else voxel_scales
    canonical = min(spatial_scales) if spatial_scales else 1.0
    factors = [s / canonical for s in spatial_scales]
    anisotropy = max(factors) / min(factors) if factors else 1.0

    info = VolumeInfo(
        bounding_box=bounding_box,
        voxel_scales=voxel_scales[:3],
        axis_names=axis_names,
        layers=layers,
        canonical_factors=factors,
        anisotropy_ratio=anisotropy,
    )

    visible_count = sum(1 for l in layers if l.visible)
    print(f"  Volume: {info.shape[0]:.0f}\u00d7{info.shape[1]:.0f}\u00d7{info.shape[2]:.0f}, "
          f"anisotropy={info.anisotropy_ratio:.1f}x, "
          f"{len(info.layers)} layers ({visible_count} visible)")
    return info


def read_shape_from_source(source_url: str) -> list[float] | None:
    """Read volume shape from a zarr source URL.

    Returns the physical extent [X, Y, Z] in the same coordinate units
    that Neuroglancer uses for positions (derived from the OME-Zarr
    multiscale transforms). This is NOT raw pixel counts.

    For an OME-Zarr with shape (T,C,Z,Y,X) = (1,1,220,1920,1920)
    and scale transform [1, 1, 1.0, 0.259, 0.259], the physical
    extent is [1920*0.259, 1920*0.259, 220*1.0] = [497, 497, 220].
    """
    url = source_url
    if url.startswith("zarr://"):
        url = url[len("zarr://"):]
    url = url.rstrip("/")

    try:
        import json as _json
        import zarr
        import s3fs

        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=url, s3=fs)
        z = zarr.open(store, mode="r")

        # Read multiscale metadata from .zattrs
        pixel_scales = None
        axes_order = None
        try:
            attrs = _json.loads(fs.cat(url + "/.zattrs"))
            if "multiscales" in attrs:
                ms = attrs["multiscales"][0]
                # Get axis names (e.g. ['t','c','z','y','x'])
                axes_order = [a["name"] for a in ms.get("axes", [])]
                # Get scale transform for highest-res dataset (path "0")
                for ds in ms.get("datasets", []):
                    if ds.get("path") == "0":
                        for t in ds.get("coordinateTransformations", []):
                            if t.get("type") == "scale":
                                pixel_scales = t["scale"]
                        break
        except Exception:
            pass

        # Get the highest-resolution array shape
        if hasattr(z, "shape"):
            voxel_shape = list(z.shape)
        elif hasattr(z, "arrays"):
            # Multiscale group: find array "0" (highest res)
            arrays = dict(z.arrays())
            if "0" in arrays:
                voxel_shape = list(arrays["0"].shape)
            elif arrays:
                _, arr = next(iter(sorted(arrays.items())))
                voxel_shape = list(arr.shape)
            else:
                return None
        else:
            return None

        # Map to spatial (X, Y, Z) in physical units
        if axes_order and pixel_scales and len(axes_order) == len(voxel_shape):
            # Use axis names to find x, y, z indices
            physical = {}
            for i, axis_name in enumerate(axes_order):
                if axis_name in ("x", "y", "z"):
                    physical[axis_name] = voxel_shape[i] * pixel_scales[i]
            if "x" in physical and "y" in physical and "z" in physical:
                result = [physical["x"], physical["y"], physical["z"]]
                print(f"  Zarr physical extent: {result[0]:.1f} × {result[1]:.1f} × {result[2]:.1f} "
                      f"(from {voxel_shape} voxels × {pixel_scales} scale)")
                return result

        # Fallback: no multiscale metadata, return raw voxel counts
        # Assume last 3 dims are Z, Y, X
        if len(voxel_shape) >= 3:
            shape_xyz = [voxel_shape[-1], voxel_shape[-2], voxel_shape[-3]]
            print(f"  Zarr voxel shape (no multiscale metadata): {shape_xyz}")
            return shape_xyz

        return None

    except Exception as e:
        print(f"  WARNING: Failed to read shape from {url}: {e}")
        return None


# ── FOV Computation ─────────────────────────────────────────────────────────

def compute_fov(scale: float, canonical_factors: list[float]) -> list[float]:
    """Compute the field of view in voxels for a given crossSectionScale.

    Returns [fov_x, fov_y, fov_z] — the number of voxels visible along
    each axis in the square viewport.
    """
    return [VIEWPORT_SIZE * scale / f for f in canonical_factors]


def compute_visible_window(
    position: list[float],
    scale: float,
    canonical_factors: list[float],
) -> list[tuple[float, float]]:
    """Compute the visible voxel window [min, max] per axis.

    Returns [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
    """
    fov = compute_fov(scale, canonical_factors)
    window = []
    for i, (pos, extent) in enumerate(zip(position[:3], fov)):
        half = extent / 2
        window.append((pos - half, pos + half))
    return window


def format_fov_feedback(
    position: list[float],
    scale: float,
    layout: str,
    volume_info: VolumeInfo,
) -> str:
    """Format FOV feedback string for inclusion in agent history."""
    window = compute_visible_window(position, scale, volume_info.canonical_factors)
    fov = compute_fov(scale, volume_info.canonical_factors)

    # Determine which axes are visible based on layout
    axis_map = {
        "xy": (0, 1), "xz": (0, 2), "yz": (1, 2),
        "3d": (0, 1), "4panel": (0, 1),
    }
    ax1, ax2 = axis_map.get(layout, (0, 1))
    names = volume_info.axis_names

    return (
        f"[visible window: "
        f"{names[ax1]}=[{window[ax1][0]:.0f}..{window[ax1][1]:.0f}], "
        f"{names[ax2]}=[{window[ax2][0]:.0f}..{window[ax2][1]:.0f}] "
        f"({fov[ax1]:.0f}\u00d7{fov[ax2]:.0f} voxels)]"
    )
