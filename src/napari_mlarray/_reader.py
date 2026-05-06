from pathlib import Path

import numpy as np
from medvol import MedVol
from mlarray import MLArray


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


def napari_get_reader(path):
    if isinstance(path, list):
        path = path[0]
    try:
        if not str(path).endswith(".mla"):
            return None
    except OSError:
        return None
    return reader_function


def _get_spatial_affine(mlarray):
    """Return the spatial affine, falling back to meta.spatial for bbox-only files."""
    affine = mlarray.affine if mlarray.affine is not None else mlarray.meta.spatial.affine
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
    return affine


def _bbox_spatial_ndim(mlarray):
    """Infer spatial ndim from bbox shape for bbox-only files."""
    if mlarray.meta.bbox.bboxes is not None:
        return np.asarray(mlarray.meta.bbox.bboxes).shape[1]
    return 3


def _bboxes_xyz_to_sarplus_voxel(bboxes_xyz, affine_xyz, mv):
    """Convert bboxes from MLArray XYZ voxel indices to SAR+ ZYX voxel indices.

    MLArray files are always produced by MedVol in canonical RAS+ form, so
    affine_xyz and mv.affine map to the same world space. The combined transform
    inv(mv.affine) @ affine_xyz converts XYZ voxels to canonical RAS+ voxels.
    SAR+ is then a simple reversal of the 3 canonical spatial axes (S=Z, A=Y, R=X).

    bboxes_xyz: (N, D, 2) — axis-1 = spatial dims in XYZ order, axis-2 = [min, max]
    Returns:    (N, D, 2) — axis-1 = spatial dims in SAR+ ZYX order, axis-2 = [min, max]
    """
    T = np.linalg.solve(mv.affine, affine_xyz)   # inv(canonical) @ original ≈ identity for standard files

    bboxes = np.asarray(bboxes_xyz, dtype=np.float64)
    N, dims, _ = bboxes.shape

    ones = np.ones((N, 1), dtype=np.float64)
    mins_h = np.hstack([bboxes[:, :, 0], ones])  # (N, D+1)
    maxs_h = np.hstack([bboxes[:, :, 1], ones])

    mins_ras = (T @ mins_h.T).T[:, :dims]  # (N, D) in canonical RAS+ voxel space
    maxs_ras = (T @ maxs_h.T).T[:, :dims]

    # SAR+ is the reverse of canonical RAS+ axis order: S=dim2, A=dim1, R=dim0
    perm = list(range(dims - 1, -1, -1))   # [2, 1, 0] for 3D, [1, 0] for 2D
    mins_sar = mins_ras[:, perm]
    maxs_sar = maxs_ras[:, perm]

    # Ensure min ≤ max (handles any flips that occurred)
    bboxes_min = np.minimum(mins_sar, maxs_sar)
    bboxes_max = np.maximum(mins_sar, maxs_sar)
    return np.stack([bboxes_min, bboxes_max], axis=-1)


def reader_function(path):
    """Read .mla file(s) and return napari LayerData tuples.

    Follows the same SAR+ display convention as napari-nifti:
      - Arrays are reoriented to SAR+ (= ZYX napari axis order).
      - scale / translate use negative spatial scales so Superior/Anterior/Right
        map to the conventional top/left corners in each 2-D slice view.
      - Bounding boxes are converted to the same SAR+ voxel coordinate space
        and share the identical scale / translate, ensuring exact alignment.
    """
    paths = [path] if isinstance(path, str) else path
    layer_data = []

    for path in paths:
        name = Path(path).stem
        mlarray = MLArray.open(path)

        affine_xyz = _get_spatial_affine(mlarray)
        has_array = mlarray.shape is not None
        has_bboxes = mlarray.meta.bbox.bboxes is not None

        if not has_array and not has_bboxes:
            continue

        spatial_ndim = mlarray.spatial_ndim if has_array else _bbox_spatial_ndim(mlarray)
        sarplus_cs = "SAR+" if spatial_ndim == 3 else "SA+"

        # Build MedVol for coordinate handling (canonicalize=True required for
        # get_array / get_geometry).  For bbox-only files use a dummy 1-voxel array.
        if has_array:
            src_array = np.asarray(mlarray)
        else:
            src_array = np.zeros((1,) * spatial_ndim, dtype=np.float32)

        if affine_xyz is not None:
            mv = MedVol(src_array, affine=affine_xyz, canonicalize=True)
        else:
            mv = MedVol(src_array, canonicalize=True)

        # SAR+ deoblique geometry: diagonal spacing / origin used for the display
        # affine (same approach as napari-nifti).
        geom = mv.get_geometry(sarplus_cs, deoblique=True)
        sp   = geom["spacing"]   # [sz, sy, sx]  always positive
        orig = geom["origin"]    # world coords of voxel (0,0,0) in SAR+ space

        # ── Image / Labels layer ───────────────────────────────────────────────
        if has_array:
            array_sar = mv.get_array(sarplus_cs)   # ZYX order
            ndim = mlarray.ndim
            sh   = array_sar.shape

            display_scale     = []
            display_translate = []
            for i in range(spatial_ndim):
                display_scale.append(-sp[i])
                display_translate.append(orig[i] + (sh[i] - 1) * sp[i])
            # Non-spatial axes (e.g. time for 4-D): positive scale, no flip.
            for i in range(spatial_ndim, ndim):
                display_scale.append(sp[i] if i < len(sp) else 1.0)
                display_translate.append(orig[i] if i < len(orig) else 0.0)

            meta = {
                "name": name,
                "scale": display_scale,
                "translate": display_translate,
                "metadata": {
                    **mlarray.meta.to_mapping(),
                    "_original_affine_xyz": (affine_xyz.tolist() if affine_xyz is not None else None),
                    "_original_coordinate_system": mv.coordinate_system,
                },
            }
            layer_type = "labels" if bool(mlarray.meta.is_seg) else "image"
            layer_data.append((array_sar, meta, layer_type))

        # ── BBox layer ────────────────────────────────────────────────────────
        if has_bboxes:
            bboxes_xyz = np.asarray(mlarray.meta.bbox.bboxes, dtype=np.float64)
            dims = bboxes_xyz.shape[1]   # number of spatial dimensions

            if affine_xyz is not None:
                bboxes_sar = _bboxes_xyz_to_sarplus_voxel(bboxes_xyz, affine_xyz, mv)
            else:
                # No affine: axis reversal is the best we can do
                bboxes_sar = bboxes_xyz[:, ::-1, :]

            labels = getattr(mlarray.meta.bbox, "labels", None)
            scores = getattr(mlarray.meta.bbox, "scores", None)

            # Bboxes share scale / translate with the image layer so they align.
            # For bbox-only files, infer virtual image extent from bbox coordinates.
            if has_array:
                bbox_scale     = display_scale[:dims]
                bbox_translate = display_translate[:dims]
                bbox_sh        = list(sh[:dims])
            else:
                max_vox    = np.max(bboxes_sar, axis=(0, 2)).astype(int) + 1
                bbox_sh    = max_vox.tolist()
                bbox_scale     = [-sp[i] for i in range(dims)]
                bbox_translate = [orig[i] + (bbox_sh[i] - 1) * sp[i] for i in range(dims)]

            if dims == 2:
                rects      = bboxes_minmax_to_napari_rectangles_2d(bboxes_sar)
                edge_color = _napari_bbox_edge_colors(rects, labels)
                text       = _napari_bbox_score_text(
                    scores, labels, len(rects), edge_color, rects
                )
                bbox_meta = {
                    "name": f"{name} (BBoxes)",
                    "shape_type": "rectangle",
                    "scale": bbox_scale,
                    "translate": bbox_translate,
                    "metadata": mlarray.meta.to_mapping(),
                    "face_color": "transparent",
                    "edge_color": edge_color,
                }
                if text is not None:
                    bbox_meta["text"] = text
                layer_data.append((rects, bbox_meta, "shapes"))

            elif dims == 3:
                box_count  = len(bboxes_sar)
                edge_color = _napari_bbox_edge_colors_count(box_count, labels)
                surface_data = bboxes_minmax_to_napari_surface_3d(bboxes_sar)
                surface_meta = {
                    "name": f"{name} (BBoxes)",
                    "scale": bbox_scale,
                    "translate": bbox_translate,
                    "metadata": mlarray.meta.to_mapping(),
                    "vertex_colors": np.repeat(
                        _with_alpha(edge_color, alpha=0.18), repeats=8, axis=0,
                    ),
                    "blending": "translucent",
                    "opacity": 1.0,
                    "shading": "flat",
                }
                layer_data.append((surface_data, surface_meta, "surface"))

            else:
                raise ValueError(
                    f"Only 2D and 3D bbox visualization is supported. Got {dims}D."
                )

    return layer_data


def bboxes_minmax_to_napari_rectangles_2d(
    bboxes,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Convert 2D bboxes in SAR+ ZYX voxel order to napari Shapes rectangles.

    Input format: (N, 2, 2) — axis-1 indexes spatial dims (dim0=Z/S, dim1=Y/A),
    axis-2 is [min, max].  Returns (N, 4, 2) corner array for napari Shapes.
    """
    arr = np.asarray(bboxes, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (2, 2):
        raise ValueError(f"Expected bboxes of shape (N, 2, 2). Got {arr.shape}.")

    # arr[:, dim, 0] = min, arr[:, dim, 1] = max
    min0 = arr[:, 0, 0]   # Z/S min
    max0 = arr[:, 0, 1]   # Z/S max
    min1 = arr[:, 1, 0]   # Y/A min
    max1 = arr[:, 1, 1]   # Y/A max

    # Guarantee ordering even if stored inverted
    min0, max0 = np.minimum(min0, max0), np.maximum(min0, max0)
    min1, max1 = np.minimum(min1, max1), np.maximum(min1, max1)

    return np.stack(
        [
            np.stack([min0, min1], axis=1),   # top-left
            np.stack([min0, max1], axis=1),   # top-right
            np.stack([max0, max1], axis=1),   # bottom-right
            np.stack([max0, min1], axis=1),   # bottom-left
        ],
        axis=1,
    ).astype(dtype, copy=False)


def bboxes_minmax_to_napari_vectors_3d(
    bboxes,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Convert 3D bboxes in SAR+ ZYX voxel order to napari Vectors (wireframe).

    Input format: (N, 3, 2) — axis-1 = ZYX spatial dims, axis-2 = [min, max].
    """
    arr = np.asarray(bboxes)
    if arr.ndim != 3 or arr.shape[1:] != (3, 2):
        raise ValueError(f"Expected bboxes of shape (N, 3, 2). Got {arr.shape}.")

    # bboxes are already in ZYX order; extract mins/maxs directly
    mins_zyx = np.minimum(arr[:, :, 0], arr[:, :, 1]).astype(dtype)
    maxs_zyx = np.maximum(arr[:, :, 0], arr[:, :, 1]).astype(dtype)

    vectors = np.empty(
        (len(arr) * len(_BBOX3D_EDGE_VERTEX_INDICES), 2, 3), dtype=dtype
    )
    cursor = 0
    for mins, maxs in zip(mins_zyx, maxs_zyx):
        corners = _bbox_corners_zyx(mins, maxs, dtype=dtype)
        for start_idx, end_idx in _BBOX3D_EDGE_VERTEX_INDICES:
            vectors[cursor, 0] = corners[start_idx]
            vectors[cursor, 1] = corners[end_idx] - corners[start_idx]
            cursor += 1
    return vectors


def bboxes_minmax_to_napari_surface_3d(
    bboxes,
    *,
    dtype=np.float32,
):
    """Convert 3D bboxes in SAR+ ZYX voxel order to a combined napari Surface mesh.

    Input format: (N, 3, 2) — axis-1 = ZYX spatial dims, axis-2 = [min, max].
    """
    arr = np.asarray(bboxes)
    if arr.ndim != 3 or arr.shape[1:] != (3, 2):
        raise ValueError(f"Expected bboxes of shape (N, 3, 2). Got {arr.shape}.")

    # bboxes are already in ZYX order; extract mins/maxs directly
    mins_zyx = np.minimum(arr[:, :, 0], arr[:, :, 1]).astype(dtype)
    maxs_zyx = np.maximum(arr[:, :, 0], arr[:, :, 1]).astype(dtype)

    vertices = np.empty((len(arr) * 8, 3), dtype=dtype)
    faces    = np.empty(
        (len(arr) * len(_BBOX3D_FACE_TRIANGLE_VERTEX_INDICES), 3), dtype=np.int32
    )
    values   = np.empty((len(arr) * 8,), dtype=dtype)

    vertex_cursor = 0
    face_cursor   = 0
    for box_index, (mins, maxs) in enumerate(zip(mins_zyx, maxs_zyx)):
        corners = _bbox_corners_zyx(mins, maxs, dtype=dtype)
        vertices[vertex_cursor : vertex_cursor + 8] = corners
        values[vertex_cursor : vertex_cursor + 8]   = box_index
        for triangle in _BBOX3D_FACE_TRIANGLE_VERTEX_INDICES:
            faces[face_cursor] = np.asarray(triangle, dtype=np.int32) + vertex_cursor
            face_cursor += 1
        vertex_cursor += 8

    return vertices, faces, values


# ── Internal geometry helpers ─────────────────────────────────────────────────

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


# ── Colour helpers ────────────────────────────────────────────────────────────

def _napari_bbox_edge_colors(rectangles, labels):
    count = len(rectangles)
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)
    if labels is not None and len(labels) == count:
        unique_labels = list(dict.fromkeys(labels))
        label_to_color = {lb: _palette_rgba(i) for i, lb in enumerate(unique_labels)}
        return np.array([label_to_color[lb] for lb in labels], dtype=np.float32)
    return np.array([_palette_rgba(i) for i in range(count)], dtype=np.float32)


def _napari_bbox_edge_colors_count(count, labels=None):
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)
    if labels is not None and len(labels) == count:
        unique_labels = list(dict.fromkeys(labels))
        label_to_color = {lb: _palette_rgba(i) for i, lb in enumerate(unique_labels)}
        return np.array([label_to_color[lb] for lb in labels], dtype=np.float32)
    return np.array([_palette_rgba(i) for i in range(count)], dtype=np.float32)


def _napari_bbox_score_text(scores, labels, count, edge_color, rectangles):
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
