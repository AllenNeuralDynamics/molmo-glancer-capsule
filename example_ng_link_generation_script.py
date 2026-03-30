"""
Utilities for generating Neuroglancer links and configurations.

This module provides functions to format neuroglancer.ViewerState objects using
format them into links for easy sharing.
"""

import json
from typing import Dict, List

import neuroglancer

DEFUALT_NG_HOST = "https://neuroglancer-demo.appspot.com/"
DEFAULT_INTENSITY_RANGE = [90, 1200]


def get_tczyx_dimension_config() -> Dict:
    """
    Get the dimension config for typical tczyx data.

    Returns
    -------
    Dict
        The dimension config to create a neuroglancer.CoordinateSpace object.
    """

    return {
        "x": [1e-06, "m"],
        "y": [1e-06, "m"],
        "z": [1e-06, "m"],
        "c": [1, ""],
        "t": [0.001, "s"],
    }


def format_shader_controls(intensity_range: List[int]) -> Dict:
    """
    Get the shader controls for a given intensity range.

    Parameters
    ----------
    intensity_range : List
        The intensity range to use. eg. [90, 700]

    Returns
    -------
    Dict
        The shader control config.
    """

    shaderControls = {"normalized": {"range": intensity_range}}

    return shaderControls


def format_shader_str(hex_str: str) -> str:
    """
    get a shader string for a given hex string color.

    Parameters
    ----------
    hex_str : str
        The hex string color to use. eg. '#FF00FF'

    Returns
    -------
    str
        The shader string.
    """

    shader = (
        '#uicontrol vec3 color color(default="'
        + hex_str
        + '")\n#uicontrol invlerp normalized\nvoid main()'
        + "{\nemitRGB(color * normalized());\n}"
    )
    return shader


def format_ng_state_link(
    viewer_state: neuroglancer.ViewerState, host: str = DEFUALT_NG_HOST
) -> str:
    """
    Format a neuroglancer state link with a given viewer state and host.

    Parameters
    ----------
    viewer_state : neuroglancer.ViewerState
        The viewer state to format into a link.
    host : str
        The host to use for the link.

    Returns
    -------
    str
        The formatted neuroglancer state link.

    """

    return f"{host}#!{json.dumps(viewer_state.to_json())}"


def get_ng_link_from_s3_json(s3_json_path):
    # format a link that uses a json file from s3

    """
    Get a neuroglancer link from a json file stored in S3.
    Parameters
    ----------
    s3_json_path : str
        The S3 path to the json file.
    Returns
    -------
    str
        The formatted neuroglancer state link.
    """
    # format link by inserting the s3 path into the neuroglancer link format
    if not s3_json_path.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")

    return f"{DEFUALT_NG_HOST}#!{s3_json_path}"

    
def get_ng_state_json_from_link(
    ng_link: str, host: str = DEFUALT_NG_HOST
) -> Dict:
    """
    Get the neuroglancer state json from a link.

    Parameters
    ----------
    ng_link : str
        The neuroglancer link to parse.
    host : str
        The host to use for the link.

    Returns
    -------
    Dict
        The neuroglancer state json.
    """

    if not ng_link.startswith(host):
        raise ValueError(f"Link must start with {host}")

    state_json = ng_link[len(host) + 2 :]  # Remove the host and '#!'
    return json.loads(state_json)


def get_viewer_state(
    layers_list: List[Dict], **kwargs
) -> neuroglancer.ViewerState:
    """
    Get a neuroglancer viewer state for a list of layers.

    Parameters
    ----------
    layers_list : List[Dict]
        A list of dicts encoding neuroglancer.Layer objects.
    **kwargs : dict
        Additional arguments passed to ViewerState.
    Returns
    -------
    neuroglancer.ViewerState
        The viewer state object.
    """

    viewer_state = neuroglancer.ViewerState(
        dimensions=neuroglancer.CoordinateSpace(get_tczyx_dimension_config()),
        layers=neuroglancer.Layers(layers_list),
        **kwargs,
    )

    return viewer_state


def get_ng_link(layers_list: List[Dict], **kwargs) -> str:
    """
    Get a neuroglancer link for a list of layers.

    Parameters
    ----------
    layers_list : List[Dict]
        A list of dicts encoding neuroglancer.Layer objects.
    **kwargs : dict
        Additional arguments passed to get_viewer_state.

    Returns
    -------
    str
        The formatted neuroglancer state link.
    """

    viewer_state = get_viewer_state(layers_list, **kwargs)
    return format_ng_state_link(viewer_state)


def get_ng_image_layer_for_zarr(
    zarr_path: str,
    name: str,
    intensity_range: List[int] = DEFAULT_INTENSITY_RANGE,
    hex_str: str = "#FF00FF",
    opacity: float = 1.0,
    blend: str = "additive",
    visible: bool = True,
    **kwargs,
) -> Dict:
    """
    Create a neuroglancer image layer for a zarr array.

    Parameters
    ----------
    zarr_path : str
        The path to the zarr array.
    name : str
        The name of the layer in the neuroglancer UI.
    intensity_range : List[int]
        The intensity range to use. eg. [90, 700]
    hex_str : str
        The hex string color to use. eg.
    opacity : float
        The opacity of the layer.
    blend : str
        The blend mode to use.
    visible : bool
        Whether the layer should be visible by default.
    **kwargs : dict
        Additional arguments passed to ImageLayer.

    Returns
    -------
    Dict
        The validated json encoding for the neuroglancer.ImageLayer object.
    """

    layer_config = {
        "name": name,
        "source": f"zarr://{zarr_path}",
        "shaderControls": format_shader_controls(intensity_range),
        "shader": format_shader_str(hex_str),
        "visible": visible,
        "opacity": opacity,
        "blend": blend,
        **kwargs,
    }

    layer = neuroglancer.ImageLayer(layer_config).to_json()

    return layer

def get_ng_segmentation_layer_for_zarr(
    zarr_path: str,
    name: str,
    hex_str: str = "#FF00FF",
    opacity: float = 1.0,
    blend: str = "additive",
    visible: bool = True,
    **kwargs,
) -> Dict:
    """
    Create a neuroglancer image layer for a zarr array.

    Parameters
    ----------
    zarr_path : str
        The path to the zarr array.
    name : str
        The name of the layer in the neuroglancer UI.
    hex_str : str
        The hex string color to use. eg.
    opacity : float
        The opacity of the layer.
    blend : str
        The blend mode to use.
    visible : bool
        Whether the layer should be visible by default.
    **kwargs : dict
        Additional arguments passed to ImageLayer.

    Returns
    -------
    Dict
        The validated json encoding for the neuroglancer.ImageLayer object.
    """

    layer_config = {
        "name": name,
        "source": f"zarr://{zarr_path}",
        "shader": format_shader_str(hex_str),
        "visible": visible,
        "opacity": opacity,
        "blend": blend,
        **kwargs,
    }

    layer = neuroglancer.SegmentationLayer(layer_config).to_json()

    return layer



def get_raw_ng_link(
    moving_path: str,
    fixed_path: str,
    moving_intensity_range: List[int] = DEFAULT_INTENSITY_RANGE,
    fixed_intensity_range: List[int] = DEFAULT_INTENSITY_RANGE,
) -> str:
    """
    Get a neuroglancer link to view raw data.

    Parameters
    ----------
    moving_path : str
        The path to the moving image zarr array.
    fixed_path : str
        The path to the fixed image zarr array.
    moving_intensity_range : List[int]
        The intensity range to use for the moving image.
    fixed_intensity_range : List[int]
        The intensity range to use for the fixed image.

    Returns
    -------
    str
        The formatted neuroglancer state link.
    """

    layers_list = [
        get_ng_image_layer_for_zarr(
            zarr_path=moving_path,
            name="moving",
            hex_str="#FF00FF",
            intensity_range=moving_intensity_range,
        ),
        get_ng_image_layer_for_zarr(
            zarr_path=fixed_path,
            name="fixed",
            hex_str="#00FF00",
            intensity_range=fixed_intensity_range,
        ),
    ]

    return get_ng_link(layers_list)


def get_alignment_ng_link(
    moving_path: str,
    fixed_path: str,
    aligned_path: str = None,
    aligned_seg_path: str = None,
    fixed_lowres: str = None,
    moving_lowres: str = None,
    moving_intensity_range: List[int] = DEFAULT_INTENSITY_RANGE,
    fixed_intensity_range: List[int] = DEFAULT_INTENSITY_RANGE,
) -> str:
    """
    Get a neuroglancer link to view alignment results.

    Parameters
    ----------
    moving_path : str
        The path to the moving image zarr array.
    fixed_path : str
        The path to the fixed image zarr array.
    aligned_path : str
        The path to the aligned image zarr array.
    moving_intensity_range : List[int]
        The intensity range to use for the moving image and aligned image.
    fixed_intensity_range : List[int]
        The intensity range to use for the fixed image.

    Returns
    -------
    str
        The formatted neuroglancer state link.
    """
    layers_list = []

    layers_list.append(
        get_ng_image_layer_for_zarr(
            zarr_path=moving_path,
            name="moving",
            hex_str="#FF00FF",
            intensity_range=moving_intensity_range,
            visible=False,
        )
    )
    layers_list.append(
        get_ng_image_layer_for_zarr(
            zarr_path=fixed_path,
            name="fixed",
            hex_str="#00FF00",
            intensity_range=fixed_intensity_range,
            visible=False,
        )
    )

    if fixed_lowres:
        layers_list.append(
            get_ng_image_layer_for_zarr(
                zarr_path=fixed_lowres,
                name="fixed_lowres",
                hex_str="#00FF00",
                intensity_range=fixed_intensity_range,
                visible=True,
            )
        )

    if moving_lowres:
        layers_list.append(
            get_ng_image_layer_for_zarr(
                zarr_path=moving_lowres,
                name="moving_lowres",
                hex_str="#FF00FF",
                intensity_range=moving_intensity_range,
                visible=False,
            )
        )
    if aligned_path:
        layers_list.append(
            get_ng_image_layer_for_zarr(
                zarr_path=aligned_path,
                name="aligned",
                hex_str="#FF00FF",
                intensity_range=moving_intensity_range,
            )
        )

    if aligned_seg_path:
        layers_list.append(
            get_ng_image_layer_for_zarr(
                zarr_path=aligned_seg_path,
                name='aligned_seg',
                hex_str='#FFFFFF',
                intensity_range=[0,1],
                opacity=0.8,
                visible=True,
            )
        )

    return get_ng_link(layers_list)


def get_all_loops_alignment_ng_link(
    final_loop_index: int,
    output_s3_directory: str,
    fixed_path: str,
    moving_path: str,
    moving_lowres: str,
    fixed_lowres: str,
    aligned_mov_zarr_name: str,
    aligned_mov_seg_zarr_name: str,
    moving_intensity_range: List[int] = DEFAULT_INTENSITY_RANGE,
    fixed_intensity_range: List[int] = DEFAULT_INTENSITY_RANGE,
) -> str:
    """
    Get a neuroglancer link to view alignment results.

    Parameters
    ----------
    moving_path : str
        The path to the moving image zarr array.
    fixed_path : str
        The path to the fixed image zarr array.
    aligned_path : str
        The path to the aligned image zarr array.
    moving_intensity_range : List[int]
        The intensity range to use for the moving image and aligned image.
    fixed_intensity_range : List[int]
        The intensity range to use for the fixed image.

    Returns
    -------
    str
        The formatted neuroglancer state link.
    """
    layers_list = []

    layers_list.append(
        get_ng_image_layer_for_zarr(
            zarr_path=fixed_path,
            name="Fixed Round Full Res",
            hex_str="#00FF00",
            intensity_range=fixed_intensity_range,
            visible=False,
        )
    )

    layers_list.append(
        get_ng_image_layer_for_zarr(
            zarr_path=moving_path,
            name="Unwarped Moving Round Full Res",
            hex_str="#FF00FF",
            intensity_range=moving_intensity_range,
            visible=False,
        )
    )

    layers_list.append(
        get_ng_image_layer_for_zarr(
            zarr_path=fixed_lowres,
            name="Fixed Round Alignment Res",
            hex_str="#00FF00",
            intensity_range=fixed_intensity_range,
            visible=True,
        )
    )

    layers_list.append(
        get_ng_image_layer_for_zarr(
            zarr_path=moving_lowres,
            name="Unwarped Moving Round Alignment Res",
            hex_str="#FF00FF",
            intensity_range=moving_intensity_range,
            visible=False,
        )
    )

    # iterate from 0 to loop_index inclusive
    for loop_index in range(0, final_loop_index + 1):
        aligned_path = f"{output_s3_directory}loop_{loop_index}/{aligned_mov_zarr_name}"
        aligned_seg_path = f"{output_s3_directory}loop_{loop_index}/{aligned_mov_seg_zarr_name}"
        
        layers_list.append(
            get_ng_image_layer_for_zarr(
                zarr_path=aligned_path,
                name="Loop " + str(loop_index) + " Aligned Moving",
                hex_str="#FF00FF",
                intensity_range=moving_intensity_range,
                visible=(loop_index == final_loop_index),
            )
        )

        layers_list.append(
            get_ng_image_layer_for_zarr(
                zarr_path=aligned_seg_path,
                name='Loop ' + str(loop_index) + ' Aligned Segmentation',
                hex_str='#FFFFFF',
                intensity_range=[0,1],
                opacity=0.75,
                visible=False,
            )
        )

    return get_ng_link(layers_list)