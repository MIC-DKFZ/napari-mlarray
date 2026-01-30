"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union
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
    mlarray = MLArray(data, meta=Meta.from_mapping(meta["metadata"]))
    mlarray.save(path)

    # return path to any file(s) that were successfully written
    return [path]
