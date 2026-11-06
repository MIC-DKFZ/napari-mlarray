"""Write napari Image and Labels layers as ``.mla`` with MLArray v1 metadata.

The geometry always comes from the layer's own transform (affine, scale, translate, rotate, shear), so a
layer that was moved in napari is saved where it is shown.

- A layer read by this plugin is written back in its original array order, with its original metadata
  (axes, values and label table, profiles, user, source, history); value scaling applied for display is
  undone when it is exact. ``ml`` statistics are dropped (the data may have been edited).
- Any other layer (painted labels, layers from other readers) gets new metadata. Its world is taken from
  ``viewer.dims.axis_labels``: when the last three are patient letters such as ``I, P, L`` (as set by this
  reader), the layer is written in the DICOM patient frame (LPS), so it lines up with the images.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from napari_mlarray._display import letters_to_lps

PATIENT_AXES = [
    {"name": "x", "unit": "millimeter", "increases_toward": "left"},
    {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
    {"name": "z", "unit": "millimeter", "increases_toward": "superior"},
]


def write_single_image(path: str, data: Any, attributes: dict) -> List[str]:
    """napari writer for one Image or Labels layer."""
    from mlarray import MLArray

    is_labels = "contrast_limits" not in attributes
    transform = _data_to_world(attributes, np.ndim(data))
    record = (attributes.get("metadata") or {}).get("napari_mlarray")
    if record and record.get("role") != "boxes":
        array, meta = _from_record(np.asarray(data), transform, record, attributes)
    else:
        array, meta = _new_metadata(np.asarray(data), transform, attributes, is_labels)
    image = MLArray(array, meta=meta)
    if not str(path).lower().endswith(".mla"):
        path = f"{path}.mla"
    image.save(path)
    return [str(path)]


def _data_to_world(attributes: dict, ndim: int) -> np.ndarray:
    """The layer's full data -> world matrix: ``affine @ (rotate * shear * scale + translate)``."""
    from napari.utils.transforms import Affine, CompositeAffine

    composite = CompositeAffine(scale=attributes.get("scale", [1] * ndim),
                                translate=attributes.get("translate", [0] * ndim),
                                rotate=attributes.get("rotate", np.eye(ndim)),
                                shear=attributes.get("shear", [0] * (ndim * (ndim - 1) // 2)), ndim=ndim)
    affine = attributes.get("affine")
    outer = Affine(affine_matrix=np.asarray(affine)) if affine is not None else Affine(ndim=ndim)
    return np.asarray(outer.affine_matrix) @ np.asarray(composite.affine_matrix)


def _from_record(data: np.ndarray, transform: np.ndarray, record: dict, attributes: dict):
    from mlarray import Meta, MLArray

    original = record.get("meta")
    if original is None:
        original = MLArray.open(record["path"]).meta
    meta_dict = copy.deepcopy(original.to_dict(include_lazy=True))
    axes = meta_dict["axes"]
    array_axis = list(record["dims"])          # display dim -> array axis (None: inserted size-1 dim)
    inserted = [d for d, a in enumerate(array_axis) if a is None]
    channel = record.get("channel")
    if record.get("true_affine") is not None:
        shown = np.asarray(record["display_affine"])
        if channel is not None:
            shown = shown[1:, 1:]
        if transform.shape == shown.shape and np.allclose(transform, shown):
            exact = np.asarray(record["true_affine"])
            transform = exact[1:, 1:] if channel is not None else exact   # not moved: write the exact geometry
    if channel is not None:
        # one channel of a split image: put the channel dim back (size 1)
        data = data[None]
        transform = np.insert(np.insert(transform, 0, 0, axis=0), 0, 0, axis=1)
        transform[0, 0] = 1
        meta_dict = _keep_channel(meta_dict, array_axis[0], channel)
        axes = meta_dict["axes"]
    n_display = data.ndim
    if n_display != len(array_axis):
        raise ValueError(f"the layer has {n_display} dims, the file it was read from {len(array_axis)}")
    n_lead = sum(1 for a in array_axis if a is not None and axes[a]["type"] != "space")

    squeeze = [d for d in inserted if data.shape[d] == 1]
    kept = [d for d in range(n_display) if d not in squeeze]
    array = data.reshape([data.shape[d] for d in kept])
    if record.get("resampled"):
        final = kept                                   # resampled grids keep the display (world) order
    else:
        final = [kept[i] for i in np.argsort([array_axis[d] for d in kept])]
        array = np.transpose(array, [kept.index(d) for d in final])
    new_axes = []
    for d in final:
        a = array_axis[d]
        new_axes.append(copy.deepcopy(axes[a]) if a is not None else {"name": f"normal_{d}", "type": "space"})
    meta_dict["axes"] = new_axes

    # geometry from the layer's transform (world rows of the space dims are the last display dims)
    if record.get("space") is not None and record.get("rotation") is not None:
        rotation = np.asarray(record["rotation"])
        n_sp = rotation.shape[0]
        rows = list(range(n_display - n_sp, n_display))
        columns = [d for d in final if d >= n_lead]
        matrix = np.zeros((n_sp + 1, len(columns) + 1))
        matrix[:n_sp, :-1] = rotation.T @ transform[np.ix_(rows, columns)]
        matrix[:n_sp, -1] = rotation.T @ transform[rows, -1]
        matrix[-1, -1] = 1.0
        names = [new_axes[final.index(d)]["name"] for d in columns]
        for space in meta_dict["spaces"]:
            if space["name"] == record["space"]:
                before = list(space["from_array"]["array_axes"])
                if sorted(before) == sorted(names):     # keep the original column order
                    pick = [names.index(n) for n in before]
                    matrix = np.hstack([matrix[:, pick], matrix[:, -1:]])
                    names = before
                space["from_array"] = {"array_axes": names, "matrix": matrix.tolist()}

    values = dict(meta_dict.get("values") or {})
    applied = record.get("applied_scale")
    if applied is not None:
        scale, offset, stored = applied
        restored = (array.astype(np.float64) - offset) / scale
        candidate = np.rint(restored).astype(stored) if np.dtype(stored).kind in "iu" else restored.astype(stored)
        if np.allclose(candidate.astype(np.float64) * scale + offset, array, rtol=0, atol=abs(scale) * 1e-3):
            array = candidate
        else:
            values.update(scale=1.0, offset=0.0)
    meta_dict["values"] = values
    ml = dict((meta_dict.get("profiles") or {}).get("ml") or {})
    if ml:
        ml.pop("statistics", None)
        changed = record.get("resampled") or tuple(array.shape) != tuple(record.get("shape", ()))
        if changed and (ml.get("boxes") or {}).get("frame") == "index":
            ml.pop("boxes")
        meta_dict["profiles"]["ml"] = ml
    for key in ("storage", "writer"):
        meta_dict.pop(key, None)
    meta = Meta.from_dict(meta_dict, shape=array.shape)
    notes = {"resampled": "oblique grid resampled for display"}
    warnings = [text for key, text in notes.items() if record.get(key)]
    meta.add_history("edited", tool=_tool(), **({"warnings": warnings} if warnings else {}))
    return np.ascontiguousarray(array), meta


def _keep_channel(meta_dict: dict, axis: int, channel: int) -> dict:
    meta_dict = copy.deepcopy(meta_dict)
    entry = meta_dict["axes"][axis]
    if entry.get("labels"):
        entry["labels"] = [entry["labels"][channel]]
    for key in ("coordinates",):
        if entry.get(key):
            entry[key] = [entry[key][channel]]
    if entry.get("start") is not None and entry.get("step"):
        entry["start"] = entry["start"] + channel * entry["step"]
    return meta_dict


def _new_metadata(data: np.ndarray, transform: np.ndarray, attributes: dict, is_labels: bool):
    from mlarray import Meta

    ndim = data.ndim
    world = _viewer_axis_labels(ndim)
    n_sp = min(ndim, 3)
    space_names = ["z", "y", "x"][-n_sp:]
    lead = [f"axis_{d}" for d in range(ndim - n_sp)]
    axes = [{"name": n, "type": "other"} for n in lead] + [{"name": n, "type": "space"} for n in space_names]
    meta_dict: Dict[str, Any] = {"axes": axes}
    rows = list(range(ndim - n_sp, ndim))
    block, shift = transform[np.ix_(rows, rows)], transform[rows, -1]
    rotation = letters_to_lps(world[-3:]) if world and n_sp == 3 else None
    if rotation is not None:
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.T @ block
        matrix[:3, 3] = rotation.T @ shift
        meta_dict["spaces"] = [{"name": "patient", "axes": PATIENT_AXES,
                                "from_array": {"array_axes": space_names, "matrix": matrix.tolist()}}]
    elif not np.allclose(transform, np.eye(ndim + 1)):
        names = [str(v) for v in (world or space_names)][-n_sp:] if world else space_names
        names = names if len(set(names)) == n_sp else space_names
        units = attributes.get("units") or [None] * ndim
        space_axes = [{"name": n, **({"unit": str(u)} if u not in (None, "pixel") else {})}
                      for n, u in zip(names, list(units)[-n_sp:])]
        matrix = np.eye(n_sp + 1)
        matrix[:n_sp, :n_sp] = block
        matrix[:n_sp, -1] = shift
        meta_dict["spaces"] = [{"name": "world", "axes": space_axes,
                                "from_array": {"array_axes": space_names, "matrix": matrix.tolist()}}]
    if is_labels:
        meta_dict["values"] = {"kind": "label", **_label_table(attributes)}
    meta = Meta(shape=data.shape, **meta_dict)
    meta.add_history("created", tool=_tool())
    return np.ascontiguousarray(data), meta


def _label_table(attributes: dict) -> Dict[str, Any]:
    features = attributes.get("features")
    if features is None or len(features) == 0 or "index" not in features or "name" not in features:
        return {}
    labels = {}
    for index, name in zip(features["index"], features["name"]):
        if name is not None and str(name):
            labels[str(int(index))] = {"name": str(name)}
    return {"labels": labels} if labels else {}


def _viewer_axis_labels(ndim: int) -> Optional[List[str]]:
    try:
        import napari

        viewer = napari.current_viewer()
    except ImportError:  # pragma: no cover
        return None
    if viewer is None:
        return None
    labels = [str(v) for v in viewer.dims.axis_labels]
    return labels[-ndim:] if len(labels) >= ndim else None


def _tool() -> str:
    from napari_mlarray import __version__

    return f"napari-mlarray {__version__}"
