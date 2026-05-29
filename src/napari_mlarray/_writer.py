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
    
    # Retrieve original XYZ affine from metadata if available
    metadata = meta["metadata"]
    original_affine = metadata.get("_original_affine_xyz")
    original_spacing = metadata.get("_original_spacing_xyz")
    original_origin = metadata.get("_original_origin_xyz")
    original_direction = metadata.get("_original_direction_xyz")
    original_coordinate_system = metadata.get("_original_coordinate_system")
    
    # Reconstruct MLArray with proper affine.
    # Strip reader-injected "_original_*" keys that are not valid Meta fields.
    meta_mapping = {k: v for k, v in metadata.items() if not k.startswith("_original_")}
    mlarray_meta = Meta.from_mapping(meta_mapping)
    if original_affine is not None:
        mlarray_meta.spatial.affine = original_affine
    elif original_spacing is not None and original_origin is not None and original_direction is not None:
        mlarray_meta.spatial.spacing = original_spacing
        mlarray_meta.spatial.origin = original_origin
        mlarray_meta.spatial.direction = original_direction
    
    if original_coordinate_system is not None:
        mlarray_meta.spatial.coord_system = original_coordinate_system
    
    mlarray = MLArray(array_xyz, meta=mlarray_meta)
    mlarray.save(path)

    # return path to any file(s) that were successfully written
    return [path]
