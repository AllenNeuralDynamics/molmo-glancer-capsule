"""
volume_info — Volume metadata discovery from Neuroglancer state + zarr sources.

Reads shape, voxel scales, layer info from the NG state and zarr .zarray metadata.
Provides FOV computation for scale-aware prompting.
"""

import json
from dataclasses import dataclass, field

VIEWPORT_SIZE = 1024


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

        lines = [
            f"Shape: {self.shape[0]} \u00d7 {self.shape[1]} \u00d7 {self.shape[2]} voxels (x \u00d7 y \u00d7 z)",
            f"Voxel size: {', '.join(scale_strs)}",
        ]
        if self.anisotropy_ratio > 1.5:
            lines.append(f"Anisotropy: z is {self.anisotropy_ratio:.1f}\u00d7 coarser than x/y")
        lines.append(f"Layers: [{', '.join(layer_strs)}]")
        lines.append(
            f"Viewport: {VIEWPORT_SIZE}\u00d7{VIEWPORT_SIZE} (square). "
            f"At crossSectionScale=S, shows S\u00b7{VIEWPORT_SIZE} \u00d7 S\u00b7{VIEWPORT_SIZE} canonical voxels."
        )
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


def read_shape_from_source(source_url: str) -> list[int] | None:
    """Read volume shape from a zarr source URL.

    Supports zarr:// and s3:// URLs. Returns [X, Y, Z] or None on failure.
    """
    # Strip zarr:// prefix
    url = source_url
    if url.startswith("zarr://"):
        url = url[len("zarr://"):]
    # Remove trailing slash
    url = url.rstrip("/")

    try:
        import zarr
        import s3fs

        # Open the zarr array at the highest resolution (level 0)
        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=url, s3=fs)
        z = zarr.open(store, mode="r")

        # zarr could be a group (multiscale) or array
        if hasattr(z, "shape"):
            shape = list(z.shape)
        elif hasattr(z, "arrays"):
            # Multiscale: get the first (highest resolution) array
            arrays = list(z.arrays())
            if arrays:
                _, arr = arrays[0]
                shape = list(arr.shape)
            else:
                return None
        else:
            return None

        # zarr shape is typically (T, C, Z, Y, X) or (Z, Y, X) or (C, Z, Y, X)
        # We need to figure out which axes are spatial
        # For NG zarr sources, shape is often just the spatial dims
        if len(shape) == 3:
            # Assume Z, Y, X order (common for zarr) → return as X, Y, Z
            return [shape[2], shape[1], shape[0]]
        elif len(shape) == 4:
            # Could be (C, Z, Y, X) or (T, Z, Y, X)
            return [shape[3], shape[2], shape[1]]
        elif len(shape) == 5:
            # (T, C, Z, Y, X)
            return [shape[4], shape[3], shape[2]]
        else:
            # Just take the last 3 dims
            return [shape[-1], shape[-2], shape[-3]]

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
