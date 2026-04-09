"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
from pathlib import Path

from mlarray import MLArray
import numpy as np


_BBOX3D_EDGE_VERTEX_INDICES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)

_BBOX3D_FACE_TRIANGLE_VERTEX_INDICES = (
    (0, 1, 3),
    (0, 3, 2),
    (4, 6, 7),
    (4, 7, 5),
    (0, 4, 5),
    (0, 5, 1),
    (2, 3, 7),
    (2, 7, 6),
    (0, 2, 6),
    (0, 6, 4),
    (1, 5, 7),
    (1, 7, 3),
)


def _get_array_zyx(mlarray):
    """Convert MLArray (XYZ) to napari ZYX order.
    
    MLArray uses canonical XYZ axis order (axis 0=X/R, 1=Y/A, 2=Z/S in RAS+).
    Napari uses ZYX order (axis 0=Z is primary slider).
    """
    array_xyz = np.asarray(mlarray)
    return np.transpose(array_xyz, (2, 1, 0))


def _bbox_mins_maxs_zyx(bboxes_xyz):
    """Return bbox mins/maxs converted from XYZ to ZYX axis order."""
    mins_xyz = np.asarray(bboxes_xyz, dtype=np.float32)[..., 0]
    maxs_xyz = np.asarray(bboxes_xyz, dtype=np.float32)[..., 1]
    mins_xyz, maxs_xyz = np.minimum(mins_xyz, maxs_xyz), np.maximum(
        mins_xyz, maxs_xyz
    )
    return mins_xyz[:, ::-1], maxs_xyz[:, ::-1]


def _bbox_corners_zyx(mins_zyx, maxs_zyx, *, dtype=np.float32):
    z0, y0, x0 = mins_zyx
    z1, y1, x1 = maxs_zyx
    return np.asarray(
        [
            [z0, y0, x0],
            [z0, y0, x1],
            [z0, y1, x0],
            [z0, y1, x1],
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x0],
            [z1, y1, x1],
        ],
        dtype=dtype,
    )


def _with_alpha(colors, alpha):
    rgba = np.asarray(colors, dtype=np.float32).copy()
    rgba[:, 3] = alpha
    return rgba


def _get_display_affine_zyx(mlarray):
    """Build display affine in ZYX order with clinical conventions.
    
    Returns a diagonal affine with negative scales for:
    - Z (dim 0): superior end at top
    - Y (dim 1): anterior end at top  
    - X (dim 2): right end at left (radiological view)
    """
    spacing = np.array(mlarray.spacing) if mlarray.spacing is not None else np.ones(mlarray.spatial_ndim)
    origin = np.array(mlarray.origin) if mlarray.origin is not None else np.zeros(mlarray.spatial_ndim)
    shape_xyz = mlarray.shape[-mlarray.spatial_ndim:]
    sx, sy, sz = spacing
    ox, oy, oz = origin
    Nx, Ny, Nz = shape_xyz
    
    affine_zyx = np.diag([-sz, -sy, -sx, 1.0])
    affine_zyx[0, 3] = oz + (Nz - 1) * sz
    affine_zyx[1, 3] = oy + (Ny - 1) * sy
    affine_zyx[2, 3] = ox + (Nx - 1) * sx
    
    return affine_zyx


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


def _spatial_affine(mlarray):
    """Return the spatial affine for any MLArray, including bbox-only files.

    ``MLArray.affine`` returns ``None`` when no array data is present even if
    ``meta.spatial.affine`` is populated, so we fall back to the metadata field.
    """
    return mlarray.affine if mlarray.affine is not None else mlarray.meta.spatial.affine


def _get_bbox_affine_zyx(mlarray):
    """Convert MLArray spatial affine (XYZ) to napari display affine (ZYX)."""
    spatial_ndim = mlarray.spatial_ndim
    if spatial_ndim is None:
        return None
    if spatial_ndim == 2:
        spacing = np.array(mlarray.spacing) if mlarray.spacing is not None else np.ones(2)
        origin = np.array(mlarray.origin) if mlarray.origin is not None else np.zeros(2)
        shape_xyz = mlarray.shape[-2:] if mlarray.shape is not None else (1, 1)
        sx, sy = spacing
        ox, oy = origin
        Nx, Ny = shape_xyz
        
        affine_zyx = np.diag([-sy, -sx, 1.0])
        affine_zyx[0, 2] = oy + (Ny - 1) * sy
        affine_zyx[1, 2] = ox + (Nx - 1) * sx
        return affine_zyx
    elif spatial_ndim >= 3:
        return _get_display_affine_zyx(mlarray)
    return None


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples."""
    paths = [path] if isinstance(path, str) else path
    layer_data = []
    for path in paths:
        name = Path(path).stem
        mlarray = MLArray.open(path)
        
        if mlarray.shape is not None:
            array_zyx = _get_array_zyx(mlarray)
            affine_zyx = _get_display_affine_zyx(mlarray)
            
            metadata = {
                "name": f"{name}",
                "affine": affine_zyx,
                "metadata": {
                    **mlarray.meta.to_mapping(),
                    "_true_affine_xyz": mlarray.affine,
                    "_true_spacing_xyz": mlarray.spacing,
                    "_true_origin_xyz": mlarray.origin,
                    "_true_direction_xyz": mlarray.direction,
                }
            }
            layer_type = "labels" if bool(mlarray.meta.is_seg) else "image"
            layer_data.append((array_zyx, metadata, layer_type))
        if mlarray.meta.bbox.bboxes is not None:
            bboxes = np.asarray(mlarray.meta.bbox.bboxes)

            # MLArray bboxes are always (N, D, 2)
            if bboxes.ndim != 3 or bboxes.shape[2] != 2:
                raise ValueError(f"Unsupported bbox shape: {bboxes.shape}")

            dims = bboxes.shape[1]
            affine = _get_bbox_affine_zyx(mlarray)

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
                    "affine": affine,
                    "metadata": mlarray.meta.to_mapping(),
                    "face_color": "transparent",
                    "edge_color": edge_color,
                }
                if text is not None:
                    metadata["text"] = text
                layer_type = "shapes"
                layer_data.append((data, metadata, layer_type))

            elif dims == 3:
                box_count = len(bboxes)
                box_edge_color = _napari_bbox_edge_colors_count(
                    count=box_count,
                    labels=getattr(mlarray.meta.bbox, "labels", None),
                )
                surface_data = bboxes_minmax_to_napari_surface_3d(bboxes)
                surface_kwargs = {
                    "name": f"{name} (BBoxes)",
                    "affine": affine,
                    "metadata": mlarray.meta.to_mapping(),
                    "vertex_colors": np.repeat(
                        _with_alpha(box_edge_color, alpha=0.18),
                        repeats=8,
                        axis=0,
                    ),
                    "blending": "translucent",
                    "opacity": 1.0,
                    "shading": "flat",
                }
                layer_data.append((surface_data, surface_kwargs, "surface"))
            else:
                raise ValueError(
                    f"Only 2D and 3D bbox visualization is supported. Got {dims}D."
                )
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


def bboxes_minmax_to_napari_vectors_3d(
    bboxes,
    *,
    dtype=np.float32,
    validate: bool = True,
) -> np.ndarray:
    """Convert 3D min/max bbox data in XYZ order to napari vectors in ZYX order."""
    arr = np.asarray(bboxes)
    if arr.ndim != 3 or arr.shape[1:] != (3, 2):
        raise ValueError(
            f"Expected bboxes of shape (N, 3, 2). Got {arr.shape}."
        )
    mins_zyx, maxs_zyx = _bbox_mins_maxs_zyx(arr)
    if validate and np.any(maxs_zyx < mins_zyx):
        bad = np.argwhere(maxs_zyx < mins_zyx)
        raise ValueError(
            "Found bbox with max < min at indices (bbox_index, dim): "
            f"{bad[:10].tolist()}" + (" ..." if len(bad) > 10 else "")
        )
    vectors = np.empty(
        (len(arr) * len(_BBOX3D_EDGE_VERTEX_INDICES), 2, 3),
        dtype=dtype,
    )
    cursor = 0
    for mins, maxs in zip(mins_zyx, maxs_zyx, strict=False):
        corners = _bbox_corners_zyx(mins, maxs, dtype=dtype)
        for start_idx, end_idx in _BBOX3D_EDGE_VERTEX_INDICES:
            start = corners[start_idx]
            end = corners[end_idx]
            vectors[cursor, 0] = start
            vectors[cursor, 1] = end - start
            cursor += 1
    return vectors


def bboxes_minmax_to_napari_surface_3d(
    bboxes,
    *,
    dtype=np.float32,
    validate: bool = True,
):
    """Convert 3D min/max bbox data in XYZ order to a combined napari surface mesh."""
    arr = np.asarray(bboxes)
    if arr.ndim != 3 or arr.shape[1:] != (3, 2):
        raise ValueError(
            f"Expected bboxes of shape (N, 3, 2). Got {arr.shape}."
        )
    mins_zyx, maxs_zyx = _bbox_mins_maxs_zyx(arr)
    if validate and np.any(maxs_zyx < mins_zyx):
        bad = np.argwhere(maxs_zyx < mins_zyx)
        raise ValueError(
            "Found bbox with max < min at indices (bbox_index, dim): "
            f"{bad[:10].tolist()}" + (" ..." if len(bad) > 10 else "")
        )

    vertices = np.empty((len(arr) * 8, 3), dtype=dtype)
    faces = np.empty(
        (len(arr) * len(_BBOX3D_FACE_TRIANGLE_VERTEX_INDICES), 3),
        dtype=np.int32,
    )
    values = np.empty((len(arr) * 8,), dtype=dtype)

    vertex_cursor = 0
    face_cursor = 0
    for box_index, (mins, maxs) in enumerate(zip(mins_zyx, maxs_zyx, strict=False)):
        corners = _bbox_corners_zyx(mins, maxs, dtype=dtype)
        vertices[vertex_cursor:vertex_cursor + 8] = corners
        values[vertex_cursor:vertex_cursor + 8] = box_index
        for triangle in _BBOX3D_FACE_TRIANGLE_VERTEX_INDICES:
            faces[face_cursor] = np.asarray(triangle, dtype=np.int32) + vertex_cursor
            face_cursor += 1
        vertex_cursor += 8
    return vertices, faces, values


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
