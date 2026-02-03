"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
from mlarray import MLArray
from pathlib import Path
import numpy as np

# Ensure napari-bbox registers its custom layer type.
import napari_bbox  # noqa: F401


def napari_get_reader(path):
    """A basic implementation of a Reader contribution."""
    if isinstance(path, list):
        path = path[0]

    try:
        if not str(path).endswith(".mla"):
            return None
    except OSError:
        return None

    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples."""
    paths = [path] if isinstance(path, str) else path
    layer_data = []
    for path in paths:
        name = Path(path).stem
        mlarray = MLArray.open(path)
        if mlarray.meta._has_array.has_array == True:
            data = mlarray
            metadata = {"name": f"{name}", "affine": mlarray.affine, "metadata": mlarray.meta.to_mapping()}
            layer_type = "labels" if mlarray.meta.is_seg.is_seg == True else "image"
            layer_data.append((data, metadata, layer_type))
        if mlarray.meta.bbox.bboxes is not None:
            bboxes = np.asarray(mlarray.meta.bbox.bboxes)

            # MLArray bboxes are always (N, D, 2)
            if bboxes.ndim != 3 or bboxes.shape[2] != 2:
                raise ValueError(f"Unsupported bbox shape: {bboxes.shape}")

            dims = bboxes.shape[1]

            # 2D -> keep shapes rectangles (original behavior)
            if dims == 2:
                data = bboxes_minmax_to_napari_rectangles_2d(bboxes)
                edge_color = _napari_bbox_edge_colors(
                    data,
                    labels=getattr(mlarray.meta.bbox, "labels", None),
                )
                text = _napari_bbox_score_text(
                    scores=getattr(mlarray.meta.bbox, "scores", None),
                    labels=getattr(mlarray.meta.bbox, "labels", None),
                    count=len(data),
                    edge_color=edge_color,
                    rectangles=data,
                )
                metadata = {
                    "name": f"{name} (BBoxes)",
                    "shape_type": "rectangle",
                    "affine": mlarray.affine,
                    "metadata": mlarray.meta.to_mapping(),
                    "face_color": "transparent",
                    "edge_color": edge_color,
                }
                if text is not None:
                    metadata["text"] = text
                layer_type = "shapes"
                layer_data.append((data, metadata, layer_type))

            # 3D+ -> napari-bbox layer
            elif dims >= 3:
                data = bboxes_minmax_to_napari_bboxes_nd(bboxes)
                edge_color = _napari_bbox_edge_colors_count(
                    count=len(data),
                    labels=getattr(mlarray.meta.bbox, "labels", None),
                )
                metadata = {
                    "name": f"{name} (BBoxes)",
                    "affine": mlarray.affine,
                    "metadata": mlarray.meta.to_mapping(),
                    "face_color": "transparent",
                    "edge_color": edge_color,
                    # "edge_width": 2,
                }
                layer_type = "boundingboxlayer"
                layer_data.append((data, metadata, layer_type))
    return layer_data


def bboxes_minmax_to_napari_rectangles_2d(
    bboxes,
    *,
    dtype=np.float32,
    validate: bool = True,
) -> np.ndarray:
    """Convert 2D axis-aligned bounding boxes from min/max format to napari Shapes rectangles."""
    arr = np.asarray(bboxes)

    if arr.ndim == 2 and arr.shape[1] == 4:
        arr = np.stack(
            [
                arr[:, [0, 2]],
                arr[:, [1, 3]],
            ],
            axis=1,
        )
    elif arr.ndim == 3 and arr.shape[1:] == (2, 2):
        pass
    else:
        raise ValueError(
            f"Expected bboxes of shape (N, 2, 2) or (N, 4). Got {arr.shape}."
        )

    # MLArray uses (N, D, 2) -> convert to (N, 2, 2)
    if arr.shape == (arr.shape[0], 2, 2):
        arr2 = arr
    else:
        arr2 = np.transpose(arr, (0, 2, 1))

    N, D, two = arr2.shape
    if D != 2 or two != 2:
        raise ValueError(f"Only 2D bboxes are supported. Got (N, {D}, {two}).")

    mins = arr2[:, 0, :]
    maxs = arr2[:, 1, :]
    # Ensure proper min/max ordering even if input is flipped
    mins, maxs = np.minimum(mins, maxs), np.maximum(mins, maxs)

    if validate and np.any(maxs < mins):
        bad = np.argwhere(maxs < mins)
        raise ValueError(
            "Found bbox with max < min at indices (bbox_index, dim): "
            f"{bad[:10].tolist()}" + (" ..." if len(bad) > 10 else "")
        )

    min0, min1 = mins[:, 0], mins[:, 1]
    max0, max1 = maxs[:, 0], maxs[:, 1]

    rects = np.stack(
        [
            np.stack([min0, min1], axis=1),
            np.stack([min0, max1], axis=1),
            np.stack([max0, max1], axis=1),
            np.stack([max0, min1], axis=1),
        ],
        axis=1,
    ).astype(dtype, copy=False)

    return rects


def bboxes_minmax_to_napari_bboxes_nd(
    bboxes,
    *,
    dtype=np.float32,
    validate: bool = True,
):
    """
    Convert N-D axis-aligned bboxes from min/max to napari-bbox format.
    Input (MLArray): (N, D, 2) where [:, :, 0] are mins and [:, :, 1] are maxs.
    Returns:
      - list of (2, D) arrays, one per bbox.
    """
    arr = np.asarray(bboxes)

    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(
            f"Expected bboxes of shape (N, D, 2). Got {arr.shape}."
        )

    mins = arr[:, :, 0]
    maxs = arr[:, :, 1]
    # Ensure proper min/max ordering even if input is flipped
    mins, maxs = np.minimum(mins, maxs), np.maximum(mins, maxs)
    if validate and np.any(maxs < mins):
        bad = np.argwhere(maxs < mins)
        raise ValueError(
            "Found bbox with max < min at indices (bbox_index, dim): "
            f"{bad[:10].tolist()}" + (" ..." if len(bad) > 10 else "")
        )

    arr2 = np.stack([mins, maxs], axis=1).astype(dtype, copy=False)
    return [arr2[i] for i in range(arr2.shape[0])]


def _napari_bbox_edge_colors(rectangles, labels):
    """Return RGBA edge colors for each bbox."""
    count = len(rectangles)
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)

    if labels is not None and len(labels) == count:
        unique_labels = list(dict.fromkeys(labels))
        label_to_color = {
            label: _palette_rgba(idx) for idx, label in enumerate(unique_labels)
        }
        colors = np.array([label_to_color[label] for label in labels], dtype=np.float32)
    else:
        colors = np.array([_palette_rgba(idx) for idx in range(count)], dtype=np.float32)

    return colors


def _napari_bbox_edge_colors_count(count, labels=None):
    """Return RGBA edge colors for each bbox (count-based)."""
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)

    if labels is not None and len(labels) == count:
        unique_labels = list(dict.fromkeys(labels))
        label_to_color = {
            label: _palette_rgba(idx) for idx, label in enumerate(unique_labels)
        }
        colors = np.array([label_to_color[label] for label in labels], dtype=np.float32)
    else:
        colors = np.array([_palette_rgba(idx) for idx in range(count)], dtype=np.float32)

    return colors


def _napari_bbox_score_text(scores, labels, count, edge_color, rectangles):
    """Return napari Shapes text metadata if scores are provided."""
    have_scores = scores is not None and len(scores) == count
    have_labels = labels is not None and len(labels) == count
    if not have_scores and not have_labels:
        return None

    top_left = rectangles[:, 0, :]
    top_left = np.maximum(top_left - np.array([4.0, 0.0], dtype=top_left.dtype), 0)

    strings = []
    for idx in range(count):
        parts = []
        if have_labels:
            parts.append(f"Label: {labels[idx]}")
        if have_scores:
            parts.append(f"Score: {scores[idx]:.3f}")
        parts.append("\n")
        strings.append("\n".join(parts))

    return {
        "string": strings,
        "color": edge_color,
        "size": 12,
        "anchor": "upper_left",
        "position": top_left,
    }


def _palette_rgba(index):
    """Simple, distinct-ish palette; returns RGBA in 0..1."""
    palette = [
        (0.90, 0.10, 0.12, 1.0),
        (0.00, 0.48, 1.00, 1.0),
        (0.20, 0.80, 0.20, 1.0),
        (0.98, 0.60, 0.00, 1.0),
        (0.60, 0.20, 0.80, 1.0),
        (0.10, 0.75, 0.80, 1.0),
        (0.80, 0.80, 0.00, 1.0),
        (0.95, 0.40, 0.60, 1.0),
        (0.90, 0.30, 0.00, 1.0),
        (0.00, 0.70, 0.40, 1.0),
        (0.40, 0.80, 1.00, 1.0),
        (1.00, 0.20, 0.70, 1.0),
        (0.50, 0.90, 0.20, 1.0),
        (0.20, 0.90, 0.70, 1.0),
        (0.70, 0.50, 1.00, 1.0),
        (1.00, 0.50, 0.20, 1.0),
        (0.20, 0.60, 1.00, 1.0),
        (1.00, 0.70, 0.20, 1.0),
        (0.60, 1.00, 0.20, 1.0),
        (0.20, 1.00, 0.40, 1.0),
        (0.20, 1.00, 0.90, 1.0),
        (0.20, 0.90, 1.00, 1.0),
        (0.40, 0.60, 1.00, 1.0),
        (0.80, 0.20, 1.00, 1.0),
        (1.00, 0.20, 0.30, 1.0),
        (1.00, 0.30, 0.50, 1.0),
        (1.00, 0.60, 0.60, 1.0),
        (1.00, 0.90, 0.30, 1.0),
        (0.60, 1.00, 0.60, 1.0),
        (0.60, 0.90, 1.00, 1.0),
    ]
    return palette[index % len(palette)]
