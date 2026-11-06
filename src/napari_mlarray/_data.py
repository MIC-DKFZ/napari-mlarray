"""Array access for napari: a lazy reordered view of an MLArray, and resampling of oblique grids."""

from __future__ import annotations

import numbers
from typing import Optional, Sequence

import numpy as np


class DisplayArray:
    """An array-like that shows ``source`` with its axes reordered and size-1 axes inserted.

    ``dims[d]`` is the source axis shown as data dim ``d``, or None for an inserted size-1 dim. Values become
    ``scale * stored + offset`` when ``scale``/``offset`` are given. Only the requested slice is read from
    ``source``, so large files stay on disk.
    """

    def __init__(self, source, dims: Sequence[Optional[int]], scale: Optional[float] = None,
                 offset: Optional[float] = None, dtype=None):
        self.source = source
        self._dims = list(dims)
        self.ndim = len(self._dims)
        self.shape = tuple(1 if axis is None else int(source.shape[axis]) for axis in self._dims)
        self.scale, self.offset = scale, offset
        self.dtype = np.dtype(dtype) if dtype is not None else np.dtype(source.dtype)
        self.size = int(np.prod(self.shape))

    def _normalize(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = next(n for n, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        return key + (slice(None),) * (self.ndim - len(key))

    def __getitem__(self, key):
        key = self._normalize(key)
        if len(key) != self.ndim or not all(isinstance(k, (slice, numbers.Integral)) for k in key):
            return np.asarray(self)[key]
        source_key = [slice(None)] * self.source.ndim
        kept = []   # data dims that stay in the result: (source axis or None)
        for d, k in enumerate(key):
            axis = self._dims[d]
            if axis is None:
                if isinstance(k, numbers.Integral):
                    if k not in (0, -1):
                        raise IndexError(f"index {k} is out of bounds for axis {d} with size 1")
                else:
                    kept.append(None if len(range(*k.indices(1))) == 1 else "empty")
                continue
            source_key[axis] = int(k) if isinstance(k, numbers.Integral) else k
            if not isinstance(k, numbers.Integral):
                kept.append(axis)
        result = np.asarray(self.source[tuple(source_key)])
        remaining = sorted(a for a in kept if isinstance(a, int))
        result = np.transpose(result, [remaining.index(a) for a in kept if isinstance(a, int)])
        for position, a in enumerate(kept):
            if a is None:
                result = np.expand_dims(result, position)
            elif a == "empty":
                result = np.expand_dims(result, position)[(slice(None),) * position + (slice(0, 0),)]
        if self.scale is not None or self.offset is not None:
            result = result.astype(self.dtype) * self.dtype.type(self.scale if self.scale is not None else 1) \
                + self.dtype.type(self.offset if self.offset is not None else 0)
        return result

    def __array__(self, dtype=None, copy=None):
        array = self[...]
        return array.astype(dtype) if dtype is not None else array

    def __len__(self):
        return self.shape[0]


def value_dtype(stored: np.dtype, scale: float, offset: float) -> np.dtype:
    """The dtype that holds ``scale * stored + offset`` without losing precision."""
    stored = np.dtype(stored)
    if stored == np.float64 or (stored.kind in "iu" and stored.itemsize >= 4):
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def resample(array: np.ndarray, n_lead: int, plan: dict, labels: bool, fill=0) -> np.ndarray:
    """Resample the trailing space dims of ``array`` onto the axis-aligned grid of ``plan``.

    ``plan`` comes from ``_display.resample_plan``: input index -> world is ``linear @ i + shift``; the
    output grid is ``origin + spacing * o``. Labels use nearest neighbour, images linear interpolation.
    """
    from scipy import ndimage

    linear, shift = np.asarray(plan["linear"]), np.asarray(plan["shift"])
    inverse = np.linalg.inv(linear)
    matrix = inverse @ np.diag(plan["spacing"])
    offset = inverse @ (np.asarray(plan["origin"]) - shift)
    lead_shape = array.shape[:n_lead]
    out_dtype = array.dtype if labels else np.result_type(array.dtype, np.float32)
    out = np.empty(lead_shape + tuple(plan["shape"]), dtype=out_dtype)
    for index in np.ndindex(*lead_shape) if lead_shape else [()]:
        out[index] = ndimage.affine_transform(array[index], matrix, offset, output_shape=tuple(plan["shape"]),
                                              order=0 if labels else 1, mode="constant", cval=fill,
                                              output=out_dtype)
    return out
