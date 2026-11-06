"""How an MLArray is placed in napari: data order, layer affine, world axes (pure numpy, no napari).

napari has no patient frame. A layer is an N-D array plus an affine from *data indices* to *world
coordinates*; the 2-D canvas shows the last two world dims of ``viewer.dims.order``, the second-to-last
running down the screen and the last running right. In 3-D the default camera is right-handed
(``('towards', 'down', 'right')``, napari >= 0.6).

Display convention (verified on screen, orientation KB ch06 section 6.3):

- A patient space (axes increasing toward left/right, anterior/posterior, superior/inferior) is shown in
  the world ``I, P, L`` (``radiological``) or ``I, P, R`` (``neurological``): each napari world dim points
  to that anatomical direction. With the default camera this gives the radiological picture in all three
  planes (axial: anterior up, patient left on the right; coronal and sagittal: superior up) and, because
  ``(I, P, L)`` is right-handed, an unmirrored 3-D rendering. ``IPR`` is left-handed: the 3-D camera depth
  has to be flipped (``camera_depth``).
- Other spaces with direction words are ordered and signed the same way: north and up point up on screen
  (world dims toward south and down), east to the right.
- Spaces without direction words keep their axes in reversed order (``z, y, x`` for ``x, y, z``), so an
  identity matrix shows the array as numpy does.
- Non-space axes (time, channel, ...) come first in the data and the world, with their sampling
  (``start``/``step``) as scale and translation.

The array is never resampled for display unless the grid is oblique to the world axes (``oblique``):
napari cannot slice such an affine in 2-D (it drops the out-of-plane rotation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: direction word -> (family, sign): family 0 = vertical/slice axis, 1 = screen-down axis, 2 = screen-right axis
_WORDS = {
    # medical (sign: +1 when the world dim points the preferred way)
    "inferior": ("si", 1.0), "superior": ("si", -1.0),
    "posterior": ("ap", 1.0), "anterior": ("ap", -1.0),
    "left": ("lr", 1.0), "right": ("lr", -1.0),
    # earth observation
    "down": ("ud", 1.0), "up": ("ud", -1.0),
    "south": ("ns", 1.0), "north": ("ns", -1.0),
    "east": ("ew", 1.0), "west": ("ew", -1.0),
}
_FAMILY_RANK = {"si": 0, "ud": 0, "ap": 1, "ns": 1, "lr": 2, "ew": 2}
_LETTER = {"inferior": "I", "superior": "S", "posterior": "P", "anterior": "A", "left": "L", "right": "R",
           "down": "D", "up": "U", "south": "S", "north": "N", "east": "E", "west": "W"}
_OPPOSITE = {"inferior": "superior", "superior": "inferior", "posterior": "anterior", "anterior": "posterior",
             "left": "right", "right": "left", "down": "up", "up": "down", "south": "north", "north": "south",
             "east": "west", "west": "east"}
#: right-handed reference frames per family set: used to tell whether the displayed world is mirrored
_LPS = {"lr": 0, "ap": 1, "si": 2}

CONVENTIONS = ("radiological", "neurological")
#: largest position change from snapping a grid to the world axes, in voxels, that still counts as axis-aligned
SNAP_TOLERANCE = 0.1
OBLIQUE = ("resample", "nearest", "keep")


@dataclass(frozen=True)
class Settings:
    """Display options of the reader.

    convention: ``radiological`` (patient left on the screen's right) or ``neurological``.
    oblique: what to do with grids that are rotated against the world axes: ``resample`` onto an
        axis-aligned grid (exact positions, interpolated values; default), ``nearest`` (snap the axes to
        the closest world axes like ITK-SNAP; positions approximate), ``keep`` (exact affine; correct in
        3-D, napari's 2-D slicing drops the out-of-plane rotation).
    """

    convention: str = "radiological"
    oblique: str = "resample"

    def __post_init__(self):
        if self.convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, got {self.convention!r}")
        if self.oblique not in OBLIQUE:
            raise ValueError(f"oblique must be one of {OBLIQUE}, got {self.oblique!r}")


@dataclass
class WorldFrame:
    """Display world of one space: ``world = rotation @ space_coordinates``."""

    rotation: np.ndarray          # (k, n_space) signed permutation
    labels: List[str]             # one per world dim (letters for direction words, else axis names)
    units: List[Optional[str]]
    words: List[Optional[str]]    # direction word of each world dim (after the sign change)
    camera_depth: str = "towards"  # 3-D camera depth that keeps the rendering unmirrored


def world_frame(space, convention: str = "radiological") -> WorldFrame:
    """The display world of ``space`` (an ``mlarray`` Space)."""
    axes = list(space.axes)
    words = [a.increases_toward for a in axes]
    known = [w in _WORDS for w in words]
    families = [_WORDS[w][0] if w in _WORDS else None for w in words]
    if all(known) and len(set(families)) == len(families):
        order = sorted(range(len(axes)), key=lambda r: _FAMILY_RANK[families[r]])
    else:
        order = list(range(len(axes)))[::-1]
    rotation = np.zeros((len(axes), len(axes)))
    labels, units, shown = [], [], []
    for row, r in enumerate(order):
        word = words[r]
        sign = 1.0
        if word in _WORDS:
            sign = _WORDS[word][1]
            if _WORDS[word][0] == "lr" and convention == "neurological":
                sign = -sign
        rotation[row, r] = sign
        shown_word = word if sign > 0 or word is None else _OPPOSITE[word]
        shown.append(shown_word)
        labels.append(_LETTER[shown_word] if shown_word in _LETTER else axes[r].name)
        units.append(axes[r].unit)
    depth = "towards"
    if len(axes) == 3 and all(f in _LPS for f in families if f) and all(known) and len(set(families)) == 3:
        # handedness of the displayed world relative to the (right-handed) LPS patient frame
        to_lps = np.zeros((3, 3))
        for r, word in enumerate(words):
            sign = 1.0 if word in ("left", "posterior", "superior") else -1.0
            to_lps[_LPS[families[r]], r] = sign
        world_in_lps = rotation @ np.linalg.inv(to_lps)
        depth = "towards" if np.linalg.det(world_in_lps) > 0 else "away"
    elif len(axes) == 3 and np.linalg.det(rotation) < 0:
        depth = "away"
    return WorldFrame(rotation, labels, units, shown, depth)


@dataclass
class Placement:
    """How one MLArray becomes napari layer data.

    ``dims[d]`` is the array axis shown as data dim ``d``, or None for a size-1 dim added so that a 2-D grid
    in a 3-D space gets a square affine. Non-space axes come first. The space dims are ordered so that data
    dim ``d`` runs along world dim ``d``: napari slices in 2-D by inverting the block of the affine that
    belongs to the displayed dims, which fails for an affine that permutes axes. ``affine`` maps data
    indices to world coordinates.
    """

    dims: List[Optional[int]]
    affine: np.ndarray
    axis_labels: List[str]
    units: List[Optional[str]]
    n_lead: int                            # number of non-space data dims (they come first)
    channel_dim: Optional[int] = None
    world: Optional[WorldFrame] = None
    space_name: Optional[str] = None
    oblique: bool = False
    resample: Optional[Dict] = None        # set when the grid must be resampled (see resample_plan)
    notes: List[str] = field(default_factory=list)
    true_affine: Optional[np.ndarray] = None   # the exact affine when ``affine`` was snapped to the world axes

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def order(self) -> List[int]:
        """The array axes in data order (without inserted dims)."""
        return [a for a in self.dims if a is not None]

    @property
    def inserted(self) -> List[int]:
        return [d for d, a in enumerate(self.dims) if a is None]


def _is_axis_aligned(linear: np.ndarray, tol: float = 1e-6) -> bool:
    columns = np.abs(linear) / np.linalg.norm(linear, axis=0, keepdims=True)
    return bool(np.all((columns < tol) | (columns > 1 - tol)))


def _snap(linear: np.ndarray) -> np.ndarray:
    """The closest signed permutation (scaled by column lengths), greedy on the largest cosine."""
    norms = np.linalg.norm(linear, axis=0)
    cos = linear / norms
    out = np.zeros_like(linear)
    used_rows, used_cols = set(), set()
    for flat in np.argsort(-np.abs(cos), axis=None):
        r, c = np.unravel_index(flat, cos.shape)
        if r in used_rows or c in used_cols:
            continue
        out[r, c] = np.sign(cos[r, c]) * norms[c]
        used_rows.add(r)
        used_cols.add(c)
    return out


def _permutation(linear: np.ndarray) -> List[int]:
    """For each world dim (row), the column that points along it most (a permutation)."""
    snapped = _snap(linear)
    return [int(np.flatnonzero(snapped[r])[0]) for r in range(linear.shape[0])]


def place(meta, shape: Sequence[int], settings: Settings = Settings(), split_channels: bool = True) -> Placement:
    """Where every array axis of an image with metadata ``meta`` (``mlarray.Meta``) and ``shape`` goes."""
    axes = list(meta.axes)
    names = [a.name for a in axes]
    space = meta.spaces[0] if meta.spaces else None
    mapped = list(space.from_array.array_axes) if space is not None else []
    space_idx = [i for i, a in enumerate(axes) if a.type == "space"]
    non_space = [i for i, a in enumerate(axes) if a.type != "space"]
    channel = None
    if split_channels:
        channels = [i for i in non_space if axes[i].type == "channel"]
        if len(channels) == 1:
            channel = channels[0]
    leading = ([channel] if channel is not None else []) + [i for i in non_space if i != channel]
    n_lead = len(leading)
    channel_dim = 0 if channel is not None else None

    labels: List[str] = []
    units: List[Optional[str]] = []
    lead_affine = np.eye(n_lead + 1)
    for d, i in enumerate(leading):
        a = axes[i]
        lead_affine[d, d] = float(a.step) if a.step else 1.0
        lead_affine[d, -1] = float(a.start) if a.start is not None else 0.0
        labels.append(a.name)
        units.append(a.unit if a.step else None)

    if space is None or not mapped:
        # index space only: identity, world dims named after the array axes
        ndim = n_lead + len(space_idx)
        affine = np.eye(ndim + 1)
        affine[:n_lead, :n_lead] = lead_affine[:-1, :-1]
        affine[:n_lead, -1] = lead_affine[:-1, -1]
        return Placement(leading + space_idx, affine, labels + [names[i] for i in space_idx],
                         units + [None] * len(space_idx), n_lead, channel_dim)

    notes: List[str] = []
    frame = world_frame(space, settings.convention)
    matrix = space.matrix()
    n_space = matrix.shape[0] - 1
    unmapped = [i for i in space_idx if names[i] not in mapped]
    if unmapped:
        notes.append(f"space axes {[names[i] for i in unmapped]} are not in space {space.name!r}; shown unscaled")
    data_space = [i for i in space_idx if names[i] in mapped]
    linear = matrix[:n_space, [mapped.index(names[i]) for i in data_space]]
    translation = matrix[:n_space, -1]
    columns: List[Optional[int]] = list(data_space)
    if len(data_space) < n_space:
        # a grid with fewer axes than the space (a single slice): add unit-length normal axes
        extra = _complete_basis(linear, n_space - len(data_space))
        linear = np.hstack([extra, linear])
        columns = [None] * extra.shape[1] + columns
        notes.append(f"{len(data_space)}-D grid in a {n_space}-D space: shown as a single slice")
    world_linear = frame.rotation @ linear
    world_shift = frame.rotation @ translation
    # data dim d runs along world dim d
    perm = _permutation(world_linear)
    world_linear = world_linear[:, perm]
    columns = [columns[c] for c in perm]

    true_linear = world_linear.copy()
    oblique = not _is_axis_aligned(world_linear)
    resample = None
    if oblique:
        # rotations far below a voxel over the whole grid (scanner round-off) count as axis-aligned
        snapped = _snap(world_linear)
        sizes = np.array([1 if c is None else int(shape[c]) for c in columns])
        corners = np.array(np.meshgrid(*[[0, max(n - 1, 0)] for n in sizes], indexing="ij")).reshape(len(sizes), -1)
        error = float(np.max(np.linalg.norm((snapped - world_linear) @ corners, axis=0)))
        if error <= SNAP_TOLERANCE * float(np.min(np.linalg.norm(world_linear, axis=0))):
            world_linear, oblique = snapped, False
            notes.append(f"grid rotated by a negligible amount: axes snapped (positions move by at most {error:.3g})")
    if oblique:
        if settings.oblique == "nearest":
            world_linear = _snap(world_linear)
            notes.append("oblique grid snapped to the closest world axes: positions are approximate")
        elif settings.oblique == "resample":
            resample = resample_plan(world_linear, world_shift,
                                     [1 if c is None else int(shape[c]) for c in columns])
            notes.append("oblique grid resampled onto world-aligned axes for display")
        else:
            notes.append("oblique grid: napari's 2-D slicing ignores the out-of-plane rotation")

    n_sp = len(columns)
    ndim = n_lead + n_sp + len(unmapped)
    affine = np.eye(ndim + 1)
    affine[:n_lead, :n_lead] = lead_affine[:-1, :-1]
    affine[:n_lead, -1] = lead_affine[:-1, -1]
    block = slice(n_lead, n_lead + n_sp)
    if resample is not None:
        affine[block, block] = np.diag(resample["spacing"])
        affine[block, -1] = resample["origin"]
    else:
        affine[block, block] = world_linear
        affine[block, -1] = world_shift
    true_affine = affine.copy()
    if resample is None:
        true_affine[block, block] = true_linear
    return Placement(leading + columns + unmapped, affine, labels + frame.labels + [names[i] for i in unmapped],
                     units + frame.units + [None] * len(unmapped), n_lead, channel_dim, frame, space.name,
                     oblique, resample, notes, true_affine)


def _complete_basis(linear: np.ndarray, count: int) -> np.ndarray:
    """``count`` unit columns orthogonal to the columns of ``linear`` (right-handed with them in 3-D)."""
    n = linear.shape[0]
    if n == 3 and linear.shape[1] == 2 and count == 1:
        normal = np.cross(linear[:, 0], linear[:, 1])
        return (normal / np.linalg.norm(normal))[:, None]
    q, _ = np.linalg.qr(np.hstack([linear, np.eye(n)]))
    return q[:, linear.shape[1]:linear.shape[1] + count]


def resample_plan(world_linear: np.ndarray, world_shift: np.ndarray, shape: Sequence[int]) -> Dict:
    """An axis-aligned output grid covering the input grid (spacing of the closest input axis per world dim)."""
    n = len(shape)
    corners = np.array(np.meshgrid(*[[0, max(s - 1, 0)] for s in shape], indexing="ij")).reshape(n, -1)
    world = world_linear @ corners + world_shift[:, None]
    cos = np.abs(world_linear) / np.linalg.norm(world_linear, axis=0, keepdims=True)
    spacing = np.linalg.norm(world_linear, axis=0)[np.argmax(cos, axis=1)]
    lo, hi = world.min(axis=1), world.max(axis=1)
    out_shape = [int(v) for v in np.floor((hi - lo) / spacing + 1e-6).astype(int) + 1]
    return {"spacing": spacing, "origin": lo, "shape": out_shape, "linear": world_linear, "shift": world_shift}


def world_to_space(points, frame: WorldFrame) -> np.ndarray:
    """Convert world coordinates (e.g. ``viewer.cursor.position[-3:]``) to the space's coordinates."""
    return np.asarray(points, float) @ frame.rotation  # rotation is orthogonal


def letters_to_lps(labels: Sequence[str]) -> Optional[np.ndarray]:
    """The rotation ``world = R @ lps`` for patient letters such as ``('I', 'P', 'L')``, else None."""
    vec = {"L": (0, 1.0), "R": (0, -1.0), "P": (1, 1.0), "A": (1, -1.0), "S": (2, 1.0), "I": (2, -1.0)}
    labels = [str(v) for v in labels]
    if len(labels) != 3 or any(v not in vec for v in labels) or sorted(vec[v][0] for v in labels) != [0, 1, 2]:
        return None
    rotation = np.zeros((3, 3))
    for row, v in enumerate(labels):
        axis, sign = vec[v]
        rotation[row, axis] = sign
    return rotation


def describe(placement: Placement) -> Tuple[str, ...]:
    """Human-readable notes on what the display changed."""
    return tuple(placement.notes)


def camera(viewer):
    """The viewer's camera (``viewer.scene.camera`` since napari 0.9, ``viewer.camera`` before)."""
    scene = getattr(viewer, "scene", None)
    return scene.camera if scene is not None and hasattr(scene, "camera") else viewer.camera


def napari_units(units) -> Optional[List[str]]:
    """Units napari (pint) understands; others become ``pixel``."""
    try:
        import pint

        registry = pint.get_application_registry()
    except ImportError:  # pragma: no cover
        return None
    out = []
    for unit in units:
        try:
            registry.Unit(unit)
            out.append(unit)
        except Exception:  # noqa: BLE001 - pint raises several error types; None also lands here
            out.append("pixel")
    return out
