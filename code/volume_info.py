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
    "region":       0.5,    # a region of the data, some edges not visible
    "neurons":      0.25,   # good for seeing and counting individual neurons
    "single-cell":  0.125,  # zoomed in on individual cells, fine detail
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
  "wide" — zoomed out, all data visible
  "full" — data fills the screen
  "region" — a region of the data
  "neurons" — see and count individual neurons
  "single-cell" — zoomed in on fine cell detail"""


@dataclass
class VolumeInfo:
    """Metadata about the volume being analyzed."""
    shape: list[int]                    # [X, Y, Z] voxel counts
    voxel_scales: list[float]           # [sx, sy, sz] in meters
    axis_names: list[str]               # ["x", "y", "z"] or ["x", "y", "z", "t"]
    layers: list[dict]                  # [{"name": ..., "type": ...}, ...]
    canonical_factors: list[float]      # [fx, fy, fz] — scale / min(scale)
    anisotropy_ratio: float             # max(factors) / min(factors)

    def format_for_prompt(self) -> str:
        """Format volume info for inclusion in the agent decision prompt."""
        # Convert voxel scales to human-readable units
        scale_strs = []
        for s, name in zip(self.voxel_scales[:3], self.axis_names[:3]):
            if s >= 1e-6:
                scale_strs.append(f"{name}={s*1e6:.1f}\u00b5m")
            else:
                scale_strs.append(f"{name}={s*1e9:.1f}nm")

        layer_strs = [f"{l['name']} ({l['type']})" for l in self.layers]

        cx = self.shape[0] / 2
        cy = self.shape[1] / 2
        cz = self.shape[2] / 2
        units = "\u00b5m"

        lines = [
            f"Size: {self.shape[0]:.0f} \u00d7 {self.shape[1]:.0f} \u00d7 {self.shape[2]:.0f} {units}",
        ]
        if self.anisotropy_ratio > 1.5:
            lines.append(f"Note: z is {self.anisotropy_ratio:.1f}\u00d7 coarser than x/y")
        lines.append(f"Layers: [{', '.join(layer_strs)}]")
        lines.append(f"Center: x={cx:.1f}, y={cy:.1f}, z={cz:.1f}")
        lines.append(f"Ranges: x=[0..{self.shape[0]:.0f}], y=[0..{self.shape[1]:.0f}], z=[0..{self.shape[2]:.0f}]")
        lines.append(f"A neuron is ~30{units} across. The scale bar in the image shows distance.")
        return "\n  ".join(lines)


def discover_volume(ng_state: dict) -> VolumeInfo:
    """Extract volume metadata from an NG state dict.

    Reads dimensions and layer info from the state. Attempts to read
    shape from zarr source if available, otherwise uses a fallback.
    """
    dims = ng_state.get("dimensions", {})
    axis_names = list(dims.keys())
    voxel_scales = [v[0] for v in dims.values()]

    layers = []
    for layer in ng_state.get("layers", []):
        layers.append({
            "name": layer.get("name", "unknown"),
            "type": layer.get("type", "image"),
        })

    # Try to read shape from the first image layer's zarr source
    shape = None
    for layer in ng_state.get("layers", []):
        source = layer.get("source", "")
        if isinstance(source, str) and "zarr" in source:
            shape = read_shape_from_source(source)
            if shape is not None:
                break

    if shape is None:
        # Fallback: try to infer from position bounds or use a default
        print("  WARNING: Could not read shape from zarr source. Using position-based estimate.")
        pos = ng_state.get("position", [0, 0, 0])
        # Use 2x position as rough estimate if position looks like center
        shape = [max(int(p * 2), 1000) for p in pos[:3]]

    # Compute canonical factors (for FOV computation)
    spatial_scales = voxel_scales[:3] if len(voxel_scales) >= 3 else voxel_scales
    canonical = min(spatial_scales) if spatial_scales else 1.0
    factors = [s / canonical for s in spatial_scales]
    anisotropy = max(factors) / min(factors) if factors else 1.0

    info = VolumeInfo(
        shape=shape[:3],
        voxel_scales=voxel_scales[:3],
        axis_names=axis_names,
        layers=layers,
        canonical_factors=factors,
        anisotropy_ratio=anisotropy,
    )
    print(f"  Volume: {info.shape[0]}\u00d7{info.shape[1]}\u00d7{info.shape[2]}, "
          f"anisotropy={info.anisotropy_ratio:.1f}x, "
          f"{len(info.layers)} layers")
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
