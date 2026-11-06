"""napari reader and writer for MLArray (``.mla``) files with correct physical geometry."""

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

from ._display import Settings, letters_to_lps, world_to_space
from ._reader import get_display, layers_from_meta, napari_get_reader, read_layers, reader_function, set_display
from ._writer import layer_to_array, write_single_image


def add_layers(viewer, layer_data):
    """Add layer data tuples (from ``read_layers``/``layers_from_meta``) to ``viewer``; returns the layers."""
    layers = []
    for data, kwargs, layer_type in layer_data:
        added = getattr(viewer, f"add_{layer_type}")(data, **kwargs)
        layers += added if isinstance(added, list) else [added]
    return layers


def add_mlarray(viewer, path, convention=None, oblique=None):
    """Add the layers of an ``.mla`` file to ``viewer``; returns the new layers.

    ``convention`` (``radiological``/``neurological``) and ``oblique`` (``resample``/``nearest``/``keep``)
    override the reader settings (``set_display``) for this file.
    """
    from ._display import Settings as _Settings

    settings = _Settings(convention or get_display().convention, oblique or get_display().oblique)
    layers = []
    for data, kwargs, layer_type in read_layers(path, settings.convention, settings.oblique):
        added = getattr(viewer, f"add_{layer_type}")(data, **kwargs)
        layers += added if isinstance(added, list) else [added]
    if settings.convention == "neurological":
        from ._display import camera

        camera(viewer).orientation = ("away", "down", "right")
    return layers


__all__ = (
    "Settings",
    "add_layers",
    "add_mlarray",
    "get_display",
    "layer_to_array",
    "layers_from_meta",
    "letters_to_lps",
    "napari_get_reader",
    "read_layers",
    "reader_function",
    "set_display",
    "world_to_space",
    "write_single_image",
)
