"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from mlarray import MLArray, Meta

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> list[str]:
    """Writes a single image layer.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """
    # Napari stores data in ZYX order (axis 0=Z).
    # Permute back to XYZ so MLArray saves in canonical orientation.
    array_xyz = np.transpose(data, (2, 1, 0))
    
    # Retrieve true XYZ affine from metadata if available
    metadata = meta["metadata"]
    true_affine = metadata.get("_true_affine_xyz")
    true_spacing = metadata.get("_true_spacing_xyz")
    true_origin = metadata.get("_true_origin_xyz")
    true_direction = metadata.get("_true_direction_xyz")
    
    # Reconstruct MLArray with proper affine
    mlarray_meta = Meta.from_mapping(metadata)
    if true_affine is not None:
        mlarray_meta.spatial.affine = true_affine
    elif true_spacing is not None and true_origin is not None and true_direction is not None:
        mlarray_meta.spatial.spacing = true_spacing
        mlarray_meta.spatial.origin = true_origin
        mlarray_meta.spatial.direction = true_direction
    
    mlarray = MLArray(array_xyz, meta=mlarray_meta)
    mlarray.save(path)

    # return path to any file(s) that were successfully written
    return [path]
