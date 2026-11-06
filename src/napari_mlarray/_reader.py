"""Read ``.mla`` files into napari layers with correct physical geometry (see ``_display`` for the convention)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from napari_mlarray import _display
from napari_mlarray._boxes import box_items, box_layers
from napari_mlarray._data import DisplayArray, resample, value_dtype

#: arrays up to this size are read into memory; larger ones are read slice by slice
LAZY_BYTES = 1 << 30

_settings = _display.Settings(
    convention=os.environ.get("NAPARI_MLARRAY_CONVENTION", "radiological"),
    oblique=os.environ.get("NAPARI_MLARRAY_OBLIQUE", "resample"),
)


def set_display(convention: Optional[str] = None, oblique: Optional[str] = None) -> _display.Settings:
    """Change how files opened from now on are displayed (see ``_display.Settings``); returns the new settings."""
    global _settings
    _settings = _display.Settings(convention or _settings.convention, oblique or _settings.oblique)
    return _settings


def get_display() -> _display.Settings:
    return _settings


def napari_get_reader(path):
    paths = path if isinstance(path, list) else [path]
    if not paths or not all(str(p).lower().endswith(".mla") for p in paths):
        return None
    return reader_function


def reader_function(path):
    paths = path if isinstance(path, list) else [path]
    layers = []
    for p in paths:
        layers += read_layers(p)
    _adjust_camera(layers)
    return layers


def read_layers(path, convention: Optional[str] = None, oblique: Optional[str] = None) -> List[tuple]:
    """napari layer data tuples ``(data, kwargs, layer_type)`` for one ``.mla`` file."""
    from mlarray import MLArray

    settings = _display.Settings(convention or _settings.convention, oblique or _settings.oblique)
    image = MLArray.open(path, mmap_mode="r" if os.name != "nt" else None)
    meta = image.meta
    name = Path(path).name[: -len(".mla")]
    shape = tuple(int(s) for s in image.shape)
    has_array = len(shape) > 0 and all(s > 0 for s in shape)
    values = meta.values
    is_labels = values.kind in ("label", "instance") and values.label_layout != "channel"
    placement = _display.place(meta, shape, settings, split_channels=not is_labels)
    for note in placement.notes:
        if not note.startswith("grid rotated by a negligible amount"):
            warnings.warn(f"{name}: {note}", stacklevel=2)
    record = {"path": str(path), "meta": meta, "dims": list(placement.dims),
              "world_labels": list(placement.axis_labels), "space": placement.space_name,
              "rotation": placement.world.rotation.tolist() if placement.world is not None else None,
              "resampled": placement.resample is not None,
              "snapped": placement.oblique and settings.oblique == "nearest",
              "convention": settings.convention, "shape": list(shape)}
    if placement.true_affine is not None and not np.allclose(placement.true_affine, placement.affine):
        # snapped for display: the writer restores the exact geometry while the layer is not moved
        record["display_affine"] = placement.affine.tolist()
        record["true_affine"] = placement.true_affine.tolist()
    layers = []
    grid_shape = None
    if has_array:
        data, kwargs, layer_type = _image_layer(image, meta, name, placement, settings, is_labels, record)
        layers.append((data, kwargs, layer_type))
        n_lead = placement.n_lead
        grid_shape = data.shape[n_lead:]
    boxes, _ = box_items(meta)
    if boxes is not None:
        if grid_shape is None:
            grid_shape, placement = _box_only_grid(meta, placement, settings)
        layers += box_layers(meta, name, placement, grid_shape, record)
    if not has_array and not layers:
        warnings.warn(f"{name}: the file has no array data and no boxes", stacklevel=2)
    return layers


def _image_layer(image, meta, name: str, placement, settings, is_labels: bool, record: Dict):
    values = meta.values
    scale = 1.0 if values.scale is None else float(values.scale)
    offset = 0.0 if values.offset is None else float(values.offset)
    apply_scale = not is_labels and values.kind == "measurement" and (scale != 1.0 or offset != 0.0)
    stored = np.dtype(image.dtype)
    dtype = value_dtype(stored, scale, offset) if apply_scale else stored
    record["applied_scale"] = [scale, offset, stored.name] if apply_scale else None
    n_lead = placement.n_lead
    nbytes = int(np.prod(image.shape)) * dtype.itemsize
    in_memory = is_labels or placement.resample is not None or nbytes <= LAZY_BYTES
    if in_memory:
        array = np.asarray(image.to_numpy())
        array = np.transpose(array, placement.order)
        for d in placement.inserted:
            array = np.expand_dims(array, d)   # increasing d: earlier insertions shift later positions right
        if apply_scale:
            array = array.astype(dtype) * dtype.type(scale) + dtype.type(offset)
        if placement.resample is not None:
            fill = values.fill if values.fill is not None else (0 if is_labels else float(np.min(array)))
            array = resample(array, n_lead, placement.resample, labels=is_labels, fill=fill)
        data = np.ascontiguousarray(array)
    else:
        data = DisplayArray(image, placement.dims, scale if apply_scale else None, offset if apply_scale else None, dtype)

    kwargs = {"name": name, "affine": placement.affine, "axis_labels": placement.axis_labels,
              "units": _display.napari_units(placement.units), "metadata": {"napari_mlarray": record}}
    if is_labels:
        if not np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.int64)
        kwargs.update(_label_kwargs(values))
        return data, kwargs, "labels"

    ml = meta.profiles.get("ml") or {}
    limits = _contrast_limits(ml.get("statistics"), scale if apply_scale else 1.0, offset if apply_scale else 0.0)
    if limits is not None:
        kwargs["contrast_limits"] = limits
    if placement.channel_dim is not None:
        channel_axis = meta.axes[placement.dims[0]]
        n = data.shape[0]
        names = list(channel_axis.labels) if channel_axis.labels and len(channel_axis.labels) == n else None
        if names is None and values.label_layout == "channel" and values.labels:
            names = [dict(values.labels).get(str(i), {}).get("name", str(i)) for i in range(n)]
        names = names or [str(i) for i in range(n)]
        drop = _drop_dim(placement.affine, 0)
        kwargs.update({"channel_axis": 0, "name": [f"{name} [{c}]" for c in names], "affine": drop,
                       "axis_labels": placement.axis_labels[1:], "units": _display.napari_units(placement.units[1:]),
                       "metadata": [{"napari_mlarray": {**record, "channel": i}} for i in range(n)]})
        if n == 1:
            kwargs["colormap"] = "gray"
    else:
        kwargs["rgb"] = False
    return data, kwargs, "image"


def _drop_dim(affine: np.ndarray, dim: int) -> np.ndarray:
    keep = [i for i in range(affine.shape[0]) if i != dim]
    return affine[np.ix_(keep, keep)]


def _contrast_limits(statistics, scale: float, offset: float):
    if not statistics:
        return None
    percentiles = statistics.get("percentiles") or {}
    low = percentiles.get("0.5", statistics.get("min"))
    high = percentiles.get("99.5", statistics.get("max"))
    if low is None or high is None:
        return None
    low, high = float(low) * scale + offset, float(high) * scale + offset
    return [low, high] if high > low else None


def _label_kwargs(values) -> Dict:
    table = {str(k): dict(v) for k, v in (values.labels or {}).items()}
    if not table:
        return {}
    keys = sorted(int(k) for k in table if k.lstrip("-").isdigit())
    features = {"index": keys, "name": [table[str(k)].get("name", "") for k in keys],
                "code": [(table[str(k)].get("code") or {}).get("meaning", "") for k in keys]}
    kwargs = {"features": features}
    if any(table[str(k)].get("color") for k in keys):
        from napari.utils.colormaps import DirectLabelColormap, label_colormap

        cyclic = label_colormap()
        color_dict = {None: (1.0, 1.0, 1.0, 1.0), 0: (0, 0, 0, 0)}
        for k in keys:
            color = table[str(k)].get("color")
            if k == 0:
                continue
            if color:
                rgb = np.asarray(color[:3], dtype=float) / 255.0
                color_dict[k] = (*rgb, 1.0)
            else:
                color_dict[k] = tuple(cyclic.map(np.array([k]))[0])
        kwargs["colormap"] = DirectLabelColormap(color_dict=color_dict)
    return kwargs


def _box_only_grid(meta, placement, settings):
    """A display grid for a file that holds boxes but no array: the index extent of the boxes."""
    boxes, items = box_items(meta)
    names = [a.name for a in meta.axes]
    if boxes["frame"] == "index":
        extent = [0] * len(names)
        for item in items:
            for name, high in zip(boxes["axes"], item["max"]):
                extent[names.index(name)] = max(extent[names.index(name)], int(np.ceil(high)) + 1)
        shape = [max(e, 1) for e in extent]
        placement = _display.place(meta, shape, _display.Settings(settings.convention, "keep"), split_channels=False)
        n_lead = placement.n_lead
        grid = [1 if a is None else shape[a] for a in placement.dims[n_lead:]]
        return grid, placement
    raise ValueError("boxes in space coordinates need the image grid; the file has no array")


def _adjust_camera(layers: List[tuple]) -> None:
    """Keep 3-D unmirrored for left-handed display worlds (neurological convention)."""
    if _settings.convention != "neurological" or not layers:
        return
    try:
        import napari

        viewer = napari.current_viewer()
    except ImportError:  # pragma: no cover
        return
    if viewer is not None:
        _display.camera(viewer).orientation = ("away", "down", "right")
