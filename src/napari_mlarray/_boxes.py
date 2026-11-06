"""Boxes of the ``ml`` profile (``profiles.ml.boxes``) as napari layers.

- 3-D boxes become a Labels layer of one-voxel box shells on the image's display grid: every slice in
  every plane shows the box outline, and the 3-D view shows the box. The array is computed per slice,
  so it costs no memory. Hovering shows the box's label and score.
- 2-D boxes become rectangles of a Shapes layer, with label and score as text.

Boxes are placed through the world: their corners (array indices for ``frame: index``, space coordinates
otherwise) are mapped to world coordinates and then to the display grid, so they align with the image
however it is displayed (reordered, flipped, resampled).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PALETTE = [
    (0.90, 0.10, 0.12), (0.00, 0.48, 1.00), (0.20, 0.80, 0.20), (0.98, 0.60, 0.00), (0.60, 0.20, 0.80),
    (0.10, 0.75, 0.80), (0.80, 0.80, 0.00), (0.95, 0.40, 0.60), (0.90, 0.30, 0.00), (0.00, 0.70, 0.40),
]


class BoxShells:
    """Lazy integer array: voxel = box number (1-based) on the shell of a box, else 0."""

    def __init__(self, shape: Sequence[int], boxes: np.ndarray):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.boxes = np.asarray(boxes, dtype=np.int64)   # (n, ndim, 2): inclusive [lo, hi] per dim
        self.dtype = np.dtype(np.uint16 if len(self.boxes) < 65535 else np.uint32)
        self.size = int(np.prod(self.shape))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = next(n for n, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (self.ndim - len(key))
        ranges, squeeze = [], []
        for d, k in enumerate(key):
            if isinstance(k, slice):
                start, stop, step = k.indices(self.shape[d])
                if step != 1:
                    return np.asarray(self)[key]
                ranges.append((start, stop))
            else:
                k = int(k) + (self.shape[d] if int(k) < 0 else 0)
                ranges.append((k, k + 1))
                squeeze.append(d)
        out = np.zeros([max(b - a, 0) for a, b in ranges], dtype=self.dtype)
        for number, box in enumerate(self.boxes, start=1):
            lo, hi = box[:, 0], box[:, 1]
            inter = [(max(a, lo[d]), min(b, hi[d] + 1)) for d, (a, b) in enumerate(ranges)]
            if any(a >= b for a, b in inter):
                continue
            for d in range(self.ndim):
                for plane in {lo[d], hi[d]}:
                    a, b = ranges[d]
                    if not a <= plane < b:
                        continue
                    region = [slice(s - ranges[e][0], t - ranges[e][0]) for e, (s, t) in enumerate(inter)]
                    region[d] = slice(plane - a, plane - a + 1)
                    out[tuple(region)] = number
        return out.squeeze(axis=tuple(squeeze)) if squeeze else out

    def __array__(self, dtype=None, copy=None):
        array = self[...]
        return array.astype(dtype) if dtype is not None else array


def box_items(meta) -> Tuple[Optional[dict], List[dict]]:
    ml = meta.profiles.get("ml") or {}
    boxes = ml.get("boxes")
    if not boxes or not boxes.get("items"):
        return None, []
    return boxes, list(boxes["items"])


def corners_to_world(boxes: dict, items: List[dict], meta, placement) -> Tuple[np.ndarray, List[str]]:
    """World coordinates of the 2^k corners of every box: (n, 2^k, world spatial dims), and the box axis names."""
    names = list(boxes["axes"])
    frame = boxes["frame"]
    lo = np.array([item["min"] for item in items], dtype=np.float64)
    hi = np.array([item["max"] for item in items], dtype=np.float64)
    k = len(names)
    signs = np.array(np.meshgrid(*[[0, 1]] * k, indexing="ij")).reshape(k, -1).T     # (2^k, k)
    corners = lo[:, None, :] * (1 - signs) + hi[:, None, :] * signs                   # (n, 2^k, k)
    n_lead = placement.n_lead
    spatial = placement.affine[n_lead:-1, n_lead:]
    if frame == "index":
        # array index (per named axis) -> data index of the displayed spatial dims -> world (unresampled grid)
        axis_names = [a.name for a in meta.axes]
        data_dims = []
        for name in names:
            array_axis = axis_names.index(name)
            if array_axis not in placement.dims:
                raise ValueError(f"box axis {name!r} is not displayed")
            d = _data_dim(placement, array_axis)
            if d < n_lead:
                raise ValueError(f"box axis {name!r} is not a space axis")
            data_dims.append(d - n_lead)
        n_sp = spatial.shape[0]
        index = np.zeros(corners.shape[:2] + (n_sp,))
        for j, d in enumerate(data_dims):
            index[..., d] = corners[..., j]
        linear, shift = _grid_affine(placement, n_lead)
        world = index @ linear.T + shift
    else:
        space = next(s for s in meta.spaces if s.name == frame)
        if placement.world is None or space.name != placement.space_name:
            raise ValueError(f"boxes in space {frame!r}, but the image is displayed in {placement.space_name!r}")
        space_axes = [a.name for a in space.axes]
        coords = np.zeros(corners.shape[:2] + (len(space_axes),))
        for j, name in enumerate(names):
            coords[..., space_axes.index(name)] = corners[..., j]
        world = coords @ placement.world.rotation.T
    return world, names


def _data_dim(placement, array_axis: int) -> int:
    return placement.dims.index(array_axis)


def _grid_affine(placement, n_lead: int) -> Tuple[np.ndarray, np.ndarray]:
    """Index -> world of the stored (not resampled) grid's spatial dims."""
    if placement.resample is not None:
        return np.asarray(placement.resample["linear"]), np.asarray(placement.resample["shift"])
    spatial = placement.affine[n_lead:-1]
    return spatial[:, n_lead:-1], spatial[:, -1]


def world_to_display_index(world: np.ndarray, placement) -> np.ndarray:
    n_lead = placement.n_lead
    block = placement.affine[n_lead:-1]
    linear, shift = block[:, n_lead:-1], block[:, -1]
    return (world - shift) @ np.linalg.inv(linear).T


def item_text(item: dict, labels: Optional[dict]) -> str:
    parts = []
    if "label" in item:
        label = item["label"]
        name = (labels or {}).get(str(label), {}).get("name") if isinstance(label, (int, np.integer)) else None
        parts.append(f"{name} ({label})" if name else str(label))
    if "score" in item:
        parts.append(f"{float(item['score']):.3f}")
    return " ".join(parts)


def class_colors(items: List[dict], labels: Optional[dict]) -> List[Tuple[float, float, float, float]]:
    """One colour per box: the label table's colour of its class, else a palette colour per class."""
    classes = list(dict.fromkeys(str(item.get("label", n)) for n, item in enumerate(items)))
    colors = []
    for n, item in enumerate(items):
        key = str(item.get("label", n))
        entry = (labels or {}).get(key) or {}
        if entry.get("color"):
            rgb = np.asarray(entry["color"][:3], dtype=float) / 255.0
            colors.append((*rgb, 1.0))
        else:
            colors.append((*PALETTE[classes.index(key) % len(PALETTE)], 1.0))
    return colors


def box_layers(meta, name: str, placement, grid_shape: Sequence[int], record: Dict) -> List[tuple]:
    """Layer data tuples for the boxes of ``meta`` on the display grid of ``placement``."""
    boxes, items = box_items(meta)
    if boxes is None:
        return []
    labels = dict(meta.values.labels) if meta.values.labels else None
    labels = {k: dict(v) for k, v in labels.items()} if labels else None
    world, axis_names = corners_to_world(boxes, items, meta, placement)
    index = world_to_display_index(world, placement)          # (n, 2^k, n_sp) display grid indices
    lo, hi = index.min(axis=1), index.max(axis=1)
    n_lead = placement.n_lead
    spatial_affine = placement.affine[n_lead:, n_lead:]
    n_sp = spatial_affine.shape[0] - 1
    texts = [item_text(item, labels) for item in items]
    colors = class_colors(items, labels)
    world_labels = placement.axis_labels[n_lead:]
    world_units = placement.units[n_lead:]
    common = {"affine": spatial_affine, "axis_labels": world_labels, "units": world_units,
              "metadata": {"napari_mlarray": {**record, "role": "boxes"}}}
    # dims a box spans (2-D boxes in a 3-D grid are flat along one dim)
    if len(axis_names) == 2 and n_sp == 2:
        rects = []
        for a, b in zip(lo, hi):
            rects.append([[a[0], a[1]], [a[0], b[1]], [b[0], b[1]], [b[0], a[1]]])
        kwargs = {**common, "name": f"{name} boxes", "shape_type": "rectangle", "face_color": "transparent",
                  "edge_color": np.array(colors), "edge_width": 1.0,
                  "features": {"text": texts}, "text": {"string": "{text}", "anchor": "upper_left", "size": 10,
                                                        "color": np.array(colors)}}
        return [(np.array(rects, dtype=np.float64), kwargs, "shapes")]
    bounds = np.stack([np.round(lo), np.round(hi)], axis=-1).astype(np.int64)
    shells = BoxShells(grid_shape, bounds)
    features = {"index": list(range(1, len(items) + 1)),
                "label": [str(item.get("label", "")) for item in items],
                "score": [float(item["score"]) if "score" in item else float("nan") for item in items],
                "box": texts}
    color_dict = {None: (0, 0, 0, 0), 0: (0, 0, 0, 0)}
    color_dict.update({n: c for n, c in enumerate(colors, start=1)})
    kwargs = {**common, "name": f"{name} boxes", "features": features, "colormap": _direct(color_dict),
              "opacity": 1.0}
    return [(shells, kwargs, "labels")]


def _direct(color_dict):
    from napari.utils.colormaps import DirectLabelColormap

    return DirectLabelColormap(color_dict=color_dict)
