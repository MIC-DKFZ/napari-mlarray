"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""
from pathlib import Path

from mlarray import MLArray
import numpy as np

from medvol.geometry import (
    SNAP_ATOL,
    CANONICAL_AXIS_LABELS,
    SIMPLEITK_AXIS_LABELS,
    CoordinateContext,
    canonical_coordinate_context,
    coordinate_system_from_affine,
    parse_coordinate_system,
    deoblique_affine,
)


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


def _parse_coordinate_system(mlarray):
    """Parse coordinate system from MLArray metadata.
    
    Returns coordinate system string (e.g., "RAS+", "LPS+", "SAR+") or None.
    """
    affine_xyz = _spatial_affine(mlarray)
    if affine_xyz is None:
        return None
    
    spatial_ndim = mlarray.spatial_ndim
    if spatial_ndim is None:
        return None
    
    axis_labels = CANONICAL_AXIS_LABELS[:spatial_ndim]
    context = CoordinateContext(
        axis_labels=axis_labels,
        anatomical_ndim=spatial_ndim,
        anatomical_axes=tuple(range(spatial_ndim)),
    )
    
    return coordinate_system_from_affine(affine_xyz, context, atol=SNAP_ATOL)


def _convert_to_coordinate_system(affine_xyz, from_cs, to_cs, shape_xyz=None):
    """Convert affine from one coordinate system to another.
    
    Args:
        affine_xyz: (N+1, N+1) affine matrix in XYZ order
        from_cs: Source coordinate system string (e.g., "RAS+", "LPS+")
        to_cs: Target coordinate system string (e.g., "SAR+")
        shape_xyz: Optional array shape for flip offset calculation
    
    Returns:
        Converted (N+1, N+1) affine in XYZ order
    """
    affine_array = np.asarray(affine_xyz, dtype=float)
    ndim = affine_array.shape[0] - 1
    
    if from_cs == to_cs:
        return affine_array.copy()
    
    from_order, from_flips = parse_coordinate_system(from_cs, ndim)
    to_order, to_flips = parse_coordinate_system(to_cs, ndim)
    
    # Build permutation matrix P that converts from_cs to to_cs
    # P maps to_cs voxel coords to from_cs voxel coords
    # P such that P @ v_to = v_from (converts to_cs coords to from_cs coords)
    P = np.zeros((ndim + 1, ndim + 1), dtype=float)
    for to_axis in range(ndim):
        from_axis = to_order[to_axis]
        P[from_axis, to_axis] = 1.0
    P[ndim, ndim] = 1.0  # Last row/col for homogeneous coordinates
    
    # P_inv is the inverse of P (transpose for permutation matrices)
    P_inv = P.T.copy()
    P_inv[ndim, ndim] = 1.0  # Ensure last diagonal is 1
    
    # Convert affine: affine_to = P^-1 @ affine_from @ P
    # affine_from @ P converts to_cs voxel coords to world via from_cs
    # P^-1 @ (...) converts the result back to to_cs voxel coords
    converted = P_inv @ affine_array @ P
    
    # Handle flips by applying flip matrices
    flip_from = np.eye(ndim + 1, dtype=float)
    flip_to = np.eye(ndim + 1, dtype=float)
    for m in range(ndim):
        if from_flips[m]:
            flip_from[m, m] = -1.0
        if to_flips[m]:
            flip_to[m, m] = -1.0
    
    # Apply flips: affine_with_flips = flip_to^-1 @ affine @ flip_from^-1
    # Since flip matrices are diagonal with ±1, their inverse equals themselves
    converted = flip_to @ converted @ flip_from
    
    # Handle flip offsets for the translation component
    if shape_xyz is not None:
        for m in range(ndim):
            if from_flips[m] and not to_flips[m]:
                converted[:, -1] += converted[:, m] * (shape_xyz[from_order[m]] - 1)
            elif not from_flips[m] and to_flips[m]:
                converted[:, -1] -= converted[:, m] * (shape_xyz[to_order[m]] - 1)
    
    return converted


def _get_sarplus_geometry_zyx(mlarray):
    """Get SAR+ geometry with deoblique affine for napari display.
    
    Returns:
        tuple: (spacing_zyx, origin_zyx, deoblique_affine_zyx)
    """
    affine_xyz = _spatial_affine(mlarray)
    if affine_xyz is None:
        spacing = np.array(mlarray.spacing) if mlarray.spacing is not None else np.ones(mlarray.spatial_ndim)
        origin = np.array(mlarray.origin) if mlarray.origin is not None else np.zeros(mlarray.spatial_ndim)
        affine_xyz = np.eye(mlarray.spatial_ndim + 1)
        affine_xyz[:-1, :-1] = np.diag(spacing)
        affine_xyz[:-1, -1] = origin
    
    shape_xyz = mlarray.shape[-mlarray.spatial_ndim:] if mlarray.shape is not None else None
    
    from_cs = _parse_coordinate_system(mlarray)
    if from_cs is None:
        from_cs = "RAS+"
    
    sarplus_affine_xyz = _convert_to_coordinate_system(affine_xyz, from_cs, "SAR+", shape_xyz)
    
    # SAR+ is already in ZYX order (S=A[2], A=Y[1], R=X[0])
    # So no need for XYZ->ZYX conversion
    deoblique_affine_xyz = deoblique_affine(sarplus_affine_xyz)
    
    # Extract spacing and origin from deoblique affine
    spacing = np.linalg.norm(deoblique_affine_xyz[:-1, :-1], axis=0)
    origin = deoblique_affine_xyz[:-1, -1]
    
    return spacing, origin, deoblique_affine_xyz
    
    spacing = np.linalg.norm(deoblique_affine_zyx[:-1, :-1], axis=0)
    origin = deoblique_affine_zyx[:-1, -1]
    
    return spacing, origin, deoblique_affine_zyx


def _convert_xyz_to_zyx_affine(affine_xyz):
    """Convert affine from XYZ to ZYX order."""
    affine_array = np.asarray(affine_xyz, dtype=float)
    ndim = affine_array.shape[0] - 1
    
    affine_zyx = np.eye(ndim + 1, dtype=float)
    
    if ndim >= 3:
        affine_zyx[0, 0] = affine_xyz[2, 2]
        affine_zyx[0, 1] = affine_xyz[2, 1]
        affine_zyx[0, 2] = affine_xyz[2, 0]
        affine_zyx[0, 3] = affine_xyz[2, 3]
        
        affine_zyx[1, 0] = affine_xyz[1, 2]
        affine_zyx[1, 1] = affine_xyz[1, 1]
        affine_zyx[1, 2] = affine_xyz[1, 0]
        affine_zyx[1, 3] = affine_xyz[1, 3]
        
        affine_zyx[2, 0] = affine_xyz[0, 2]
        affine_zyx[2, 1] = affine_xyz[0, 1]
        affine_zyx[2, 2] = affine_xyz[0, 0]
        affine_zyx[2, 3] = affine_xyz[0, 3]
    elif ndim == 2:
        affine_zyx[0, 0] = affine_xyz[1, 1]
        affine_zyx[0, 1] = affine_xyz[1, 0]
        affine_zyx[0, 2] = affine_xyz[1, 2]
        
        affine_zyx[1, 0] = affine_xyz[0, 1]
        affine_zyx[1, 1] = affine_xyz[0, 0]
        affine_zyx[1, 2] = affine_xyz[0, 2]
    
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
    affine = mlarray.affine if mlarray.affine is not None else mlarray.meta.spatial.affine
    if affine is not None:
        affine = np.asarray(affine)
    return affine


def _get_bbox_affine_zyx(mlarray):
    """Get SAR+ affine for napari display with negative scales."""
    # Use medvol to get SAR+ geometry directly
    # This handles coordinate system conversion correctly
    from medvol import MedVol
    
    # Get spatial_ndim (may be None for bbox-only files)
    if mlarray.spatial_ndim is None:
        spatial_ndim = 2  # default to 2D for bbox-only files
    else:
        spatial_ndim = mlarray.spatial_ndim
    
    # Determine coordinate system based on spatial_ndim
    if spatial_ndim == 2:
        sarplus_cs = "SA+"
    else:
        sarplus_cs = "SAR+"
    
    # Build MedVol with the MLArray's affine
    affine = _spatial_affine(mlarray)
    if affine is None:
        # For bbox-only files, try to get spatial affine from meta
        if hasattr(mlarray, 'meta') and hasattr(mlarray.meta, 'spatial') and mlarray.meta.spatial is not None:
            affine = mlarray.meta.spatial.affine
        if affine is None:
            # Create identity affine based on inferred spatial_ndim
            affine = np.eye(spatial_ndim + 1)
    
    # Get shape (may be None for bbox-only files)
    if mlarray.shape is not None and mlarray.spatial_ndim is not None:
        shape = mlarray.shape[-mlarray.spatial_ndim:]
    else:
        shape = None
    
    # Create MedVol and get SAR+ geometry
    # Use shape to create zero array, or fallback to (1,1,1)
    if shape is not None:
        array_for_medvol = np.zeros(shape, dtype=np.float32)
    else:
        # Infer shape from bbox if available
        if hasattr(mlarray, 'meta') and hasattr(mlarray.meta, 'bbox') and mlarray.meta.bbox is not None:
            if hasattr(mlarray.meta.bbox, 'get'):
                bboxes = mlarray.meta.bbox.get('bboxes', [])
            else:
                bboxes = getattr(mlarray.meta.bbox, 'bboxes', None) or []
            if len(bboxes) > 0:
                bbox_array = np.asarray(bboxes)
                D = bbox_array.shape[1] if bbox_array.ndim >= 2 else spatial_ndim
                if D == 3:
                    array_for_medvol = np.zeros((1, 1, 1), dtype=np.float32)
                elif D == 2:
                    array_for_medvol = np.zeros((1, 1), dtype=np.float32)
                else:
                    array_for_medvol = np.zeros((1,) * D, dtype=np.float32)
            else:
                array_for_medvol = np.zeros((1,) * spatial_ndim, dtype=np.float32)
        else:
            array_for_medvol = np.zeros((1,) * spatial_ndim, dtype=np.float32)
    
    medvol = MedVol(
        array_for_medvol,
        affine=affine,
        canonicalize=True
    )
    
    geom = medvol.get_geometry(sarplus_cs, deoblique=True)
    
    # Extract spacing and origin in SAR+ order
    spacing = geom["spacing"]  # For 3D: [sz, sy, sx], For 2D: [sy, sx]
    origin = geom["origin"]    # For 3D: [oz, oy, ox], For 2D: [oy, ox]
    
    # Get shape in SAR+ order
    # MLArray shape is in XYZ order [X, Y, Z], permute to SAR+ [Z, Y, X]
    if shape is not None:
        if spatial_ndim == 3:
            shape_sar = (shape[2], shape[1], shape[0])
        else:  # spatial_ndim == 2
            shape_sar = (shape[1], shape[0])
    else:
        # Use inferred shape from array_for_medvol
        shape_sar = array_for_medvol.shape[-spatial_ndim:]
        if spatial_ndim == 2:
            shape_sar = (shape_sar[1], shape_sar[0])  # [Y, X] -> [X, Y] then permute
        else:
            shape_sar = (shape_sar[2], shape_sar[1], shape_sar[0])
    
    # Build napari affine with negative scales
    if spatial_ndim >= 3:
        # SAR+ spacing: [sz, sy, sx] = [S, A, R]
        # SAR+ origin: [oz, oy, ox] = [S, A, R]
        # Shape: [Nz, Ny, Nx] = [S, A, R] dimensions
        Nz, Ny, Nx = shape_sar[0], shape_sar[1], shape_sar[2]
        
        # napari expects ZYX order with negative scales
        # Z=S, Y=A, X=R, so we use spacing and origin directly
        affine_zyx = np.diag([-spacing[0], -spacing[1], -spacing[2], 1.0])
        affine_zyx[0, 3] = origin[0] + (Nz - 1) * spacing[0]
        affine_zyx[1, 3] = origin[1] + (Ny - 1) * spacing[1]
        affine_zyx[2, 3] = origin[2] + (Nx - 1) * spacing[2]
        return affine_zyx
    elif spatial_ndim == 2:
        # 2D SAR+ spacing: [sy, sx] = [S, A]
        # 2D SAR+ origin: [oy, ox] = [S, A]
        # Shape: [Ny, Nx] = [S, A] dimensions
        Ny, Nx = shape_sar[0], shape_sar[1]
        
        # napari expects ZYX order (with Z padded), negative scales for first 2 dims
        # Z=S, Y=A, so we use spacing and origin directly for Y and X
        affine_zyx = np.diag([-spacing[0], -spacing[1], 1.0, 1.0])
        affine_zyx[0, 2] = origin[0] + (Ny - 1) * spacing[0]
        affine_zyx[1, 2] = origin[1] + (Nx - 1) * spacing[1]
        return affine_zyx
    return None


def _get_array_zyx_with_sarplus(mlarray):
    """Convert MLArray (XYZ) to napari ZYX order with SAR+ coordinate system.
    
    Returns the array in ZYX order with SAR+ orientation.
    """
    array_xyz = np.asarray(mlarray)
    from_cs = _parse_coordinate_system(mlarray)
    if from_cs is None:
        from_cs = "RAS+"
    
    if mlarray.spatial_ndim == 2:
        sarplus_cs = "SA+"
    else:
        sarplus_cs = "SAR+"
    
    if from_cs == sarplus_cs:
        return np.transpose(array_xyz, (2, 1, 0))
    
    from_order, from_flips = parse_coordinate_system(from_cs, mlarray.spatial_ndim)
    to_order, to_flips = parse_coordinate_system(sarplus_cs, mlarray.spatial_ndim)
    
    full_axis_order = list(to_order) + list(range(mlarray.spatial_ndim, mlarray.ndim))
    array_converted = np.transpose(array_xyz, full_axis_order)
    
    for m, flip in enumerate(to_flips):
        if flip:
            array_converted = np.flip(array_converted, axis=m)
    
    return np.transpose(array_converted, (2, 1, 0))


def _convert_bboxes_to_sarplus(bboxes_xyz, mlarray):
    """Convert bboxes from their original coordinate system to SAR+.
    
    MLArray stores bboxes in (N, D, 2) format where:
    - N = number of bboxes
    - D = spatial dimensions  
    - 2 = [min, max] values
    
    Axis 1 is spatial dimension order (0=X, 1=Y, 2=Z for 3D)
    Axis 2 is [min, max] values
    
    After conversion, bboxes are still in (N, D, 2) format but with SAR+ coordinates.
    """
    from_cs = _parse_coordinate_system(mlarray)
    if from_cs is None:
        from_cs = "RAS+"
    
    if mlarray.spatial_ndim == 2:
        sarplus_cs = "SA+"
    else:
        sarplus_cs = "SAR+"
    
    if from_cs == sarplus_cs:
        return bboxes_xyz
    
    affine_xyz = _spatial_affine(mlarray)
    if affine_xyz is None:
        return bboxes_xyz
    
    shape_xyz = mlarray.shape[-mlarray.spatial_ndim:] if mlarray.shape is not None else None
    sarplus_affine_xyz = _convert_to_coordinate_system(affine_xyz, from_cs, sarplus_cs, shape_xyz)
    
    bboxes_array = np.asarray(bboxes_xyz, dtype=np.float32)
    
    ndim = mlarray.spatial_ndim
    if ndim == 2:
        mins_xyz = bboxes_array[:, :, 0]
        maxs_xyz = bboxes_array[:, :, 1]
        
        mins_h = np.hstack([mins_xyz, np.ones((len(mins_xyz), 1), dtype=np.float32)]).T
        maxs_h = np.hstack([maxs_xyz, np.ones((len(maxs_xyz), 1), dtype=np.float32)]).T
        
        mins_sar_xyz = sarplus_affine_xyz @ mins_h
        maxs_sar_xyz = sarplus_affine_xyz @ maxs_h
        
        mins_sar_xyz = mins_sar_xyz[:2, :].T
        maxs_sar_xyz = maxs_sar_xyz[:2, :].T
        
        bboxes_sar = np.stack([mins_sar_xyz, maxs_sar_xyz], axis=2)
        
        return bboxes_sar
    elif ndim == 3:
        mins_xyz = bboxes_array[:, :, 0]
        maxs_xyz = bboxes_array[:, :, 1]
        
        mins_h = np.hstack([mins_xyz, np.ones((len(mins_xyz), 1), dtype=np.float32)]).T
        maxs_h = np.hstack([maxs_xyz, np.ones((len(maxs_xyz), 1), dtype=np.float32)]).T
        
        mins_sar_xyz = sarplus_affine_xyz @ mins_h
        maxs_sar_xyz = sarplus_affine_xyz @ maxs_h
        
        mins_sar_xyz = mins_sar_xyz[:3, :].T
        maxs_sar_xyz = maxs_sar_xyz[:3, :].T
        
        bboxes_sar = np.stack([mins_sar_xyz, maxs_sar_xyz], axis=2)
        
        return bboxes_sar
    
    return bboxes_xyz


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples."""
    paths = [path] if isinstance(path, str) else path
    layer_data = []
    for path in paths:
        name = Path(path).stem
        mlarray = MLArray.open(path)
        
        coordinate_system = _parse_coordinate_system(mlarray)
        if coordinate_system is None:
            coordinate_system = "RAS+"
        
        if mlarray.shape is not None:
            array_zyx = _get_array_zyx_with_sarplus(mlarray)
            
            spacing_zyx, origin_zyx, deoblique_affine_zyx = _get_sarplus_geometry_zyx(mlarray)
            
            ndim = mlarray.ndim
            display_scale = []
            display_translate = []
            shape_zyx = array_zyx.shape[:ndim]
            
            for i in range(3):
                display_scale.append(-spacing_zyx[i])
                display_translate.append(origin_zyx[i] + (shape_zyx[i] - 1) * spacing_zyx[i])
            
            for i in range(3, ndim):
                display_scale.append(spacing_zyx[i] if i < len(spacing_zyx) else 1.0)
                display_translate.append(origin_zyx[i] if i < len(origin_zyx) else 0.0)
            
            metadata = {
                "name": f"{name}",
                "scale": display_scale,
                "translate": display_translate,
                "metadata": {
                    **mlarray.meta.to_mapping(),
                    "_original_affine_xyz": mlarray.affine,
                    "_original_spacing_xyz": mlarray.spacing,
                    "_original_origin_xyz": mlarray.origin,
                    "_original_direction_xyz": mlarray.direction,
                    "_original_coordinate_system": coordinate_system,
                }
            }
            layer_type = "labels" if bool(mlarray.meta.is_seg) else "image"
            layer_data.append((array_zyx, metadata, layer_type))
        if mlarray.meta.bbox.bboxes is not None:
            bboxes_xyz = np.asarray(mlarray.meta.bbox.bboxes)

            if bboxes_xyz.ndim != 3 or bboxes_xyz.shape[2] != 2:
                raise ValueError(f"Unsupported bbox shape: {bboxes_xyz.shape}")

            dims = bboxes_xyz.shape[1]
            
            bboxes_sar = _convert_bboxes_to_sarplus(bboxes_xyz, mlarray)
            
            deoblique_affine_zyx = _get_bbox_affine_zyx(mlarray)

            if dims == 2:
                data = bboxes_minmax_to_napari_rectangles_2d(bboxes_sar)
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
                    "affine": deoblique_affine_zyx,
                    "metadata": mlarray.meta.to_mapping(),
                    "face_color": "transparent",
                    "edge_color": edge_color,
                }
                if text is not None:
                    metadata["text"] = text
                layer_type = "shapes"
                layer_data.append((data, metadata, layer_type))

            elif dims == 3:
                box_count = len(bboxes_sar)
                box_edge_color = _napari_bbox_edge_colors_count(
                    count=box_count,
                    labels=getattr(mlarray.meta.bbox, "labels", None),
                )
                surface_data = bboxes_minmax_to_napari_surface_3d(bboxes_sar)
                surface_kwargs = {
                    "name": f"{name} (BBoxes)",
                    "affine": deoblique_affine_zyx,
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
