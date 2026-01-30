"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
from mlarray import MLArray
from pathlib import Path
import numpy as np


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # the get_reader function should make as many checks as possible
    # (without loading the full file) to determine if it can read
    # the path. Here, we check the dtype of the array by loading
    # it with memmap, so that we don't actually load the full array into memory.
    # We pretend that this reader can only read integer arrays.
    try:
        if not str(path).endswith(".mla"):
            return None
    # napari_get_reader should never raise an exception, because napari
    # raises its own specific errors depending on what plugins are
    # available for the given path, so we catch
    # the OSError that np.load might raise if the file is malformed
    except OSError:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
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
            data = bboxes_minmax_to_napari_rectangles_2d(mlarray.meta.bbox.bboxes)
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
    return layer_data


def bboxes_minmax_to_napari_rectangles_2d(
    bboxes,
    *,
    dtype=np.float32,
    validate: bool = True,
) -> np.ndarray:
    """
    Convert 2D axis-aligned bounding boxes from min/max format to napari Shapes rectangles.

    Accepted input formats (both mean the same thing):
      1) (N, 2, 2): [[min_dim0, max_dim0], [min_dim1, max_dim1]]
         Example (dim order is whatever you use, e.g. (y, x)):
           [[[ymin, ymax], [xmin, xmax]], ...]

      2) (N, 4): [min_dim0, min_dim1, max_dim0, max_dim1]
         Example:
           [[ymin, xmin, ymax, xmax], ...]

    Output format (napari Shapes rectangle vertices):
      (N, 4, 2) with vertices in non-twisting cyclic order:
        (min0, min1) -> (min0, max1) -> (max0, max1) -> (max0, min1)

    Raises:
      ValueError if bboxes are not 2D (i.e., D != 2) or shapes are invalid.
    """
    arr = np.asarray(bboxes)

    # Normalize input to shape (N, 2, 2)
    if arr.ndim == 2 and arr.shape[1] == 4:
        # (N, 4) -> (N, 2, 2)
        arr = np.stack(
            [
                arr[:, [0, 2]],  # dim0: [min0, max0]
                arr[:, [1, 3]],  # dim1: [min1, max1]
            ],
            axis=1,
        )
    elif arr.ndim == 3 and arr.shape[1:] == (2, 2):
        pass
    else:
        raise ValueError(
            f"Expected bboxes of shape (N, 2, 2) or (N, 4). Got {arr.shape}."
        )

    N, D, two = arr.shape
    if D != 2 or two != 2:
        # Defensive; should never hit because of checks above.
        raise ValueError(f"Only 2D bboxes are supported. Got (N, {D}, {two}).")

    mins = arr[:, :, 0]
    maxs = arr[:, :, 1]

    if validate and np.any(maxs < mins):
        bad = np.argwhere(maxs < mins)
        raise ValueError(
            "Found bbox with max < min at indices (bbox_index, dim): "
            f"{bad[:10].tolist()}" + (" ..." if len(bad) > 10 else "")
        )

    min0, min1 = mins[:, 0], mins[:, 1]
    max0, max1 = maxs[:, 0], maxs[:, 1]

    # Cyclic order (no twisting):
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


def _napari_bbox_score_text(scores, labels, count, edge_color, rectangles):
    """Return napari Shapes text metadata if scores are provided."""
    have_scores = scores is not None and len(scores) == count
    have_labels = labels is not None and len(labels) == count
    if not have_scores and not have_labels:
        return None

    # Place text at the top-left corner of each rectangle.
    top_left = rectangles[:, 0, :]
    top_left = np.maximum(top_left - np.array([4.0, 0.0], dtype=top_left.dtype), 0)

    strings = []
    for idx in range(count):
        parts = []
        if have_labels:
            parts.append(f"Label: {labels[idx]}")
        if have_scores:
            parts.append(f"Score: {scores[idx]:.3f}")
        # Add a trailing empty line to create spacing below the score.
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
