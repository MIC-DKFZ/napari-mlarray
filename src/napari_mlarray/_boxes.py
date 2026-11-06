"""Boxes of the ``ml`` profile (``profiles.ml.boxes``) as napari layers, drawn by napari-bboxes.

Boxes are placed through the world: their corners (array indices for ``frame: index``, space coordinates
otherwise) are mapped to world coordinates and then to the image's display grid, so they align with the image
however it is displayed (reordered, flipped, resampled). napari-bboxes then draws them: 3-D boxes as outlines
visible in every plane and in 3-D, 2-D boxes as rectangles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from napari_mlarray._display import napari_units


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


def box_layers(meta, name: str, placement, grid_shape: Sequence[int], record: Dict) -> List[tuple]:
    """Layer data tuples for the boxes of ``meta`` on the display grid of ``placement``."""
    from napari_bboxes import bbox_layers

    boxes, items = box_items(meta)
    if boxes is None:
        return []
    table = {str(k): dict(v) for k, v in (meta.values.labels or {}).items()}
    world, _ = corners_to_world(boxes, items, meta, placement)
    index = world_to_display_index(world, placement)          # (n, 2^k, n_sp) display grid indices
    n_lead = placement.n_lead
    labels = [item.get("label", "") for item in items]
    return bbox_layers(
        index.min(axis=1), index.max(axis=1),
        labels=labels,
        scores=[item.get("score") for item in items] if any("score" in item for item in items) else None,
        names={k: v["name"] for k, v in table.items() if v.get("name")},
        colors={k: v["color"] for k, v in table.items() if v.get("color")},
        grid_shape=grid_shape,
        affine=placement.affine[n_lead:, n_lead:],
        name=f"{name} boxes",
        axis_labels=placement.axis_labels[n_lead:],
        units=napari_units(placement.units[n_lead:]),
        metadata={"napari_mlarray": {**record, "role": "boxes"}},
    )
