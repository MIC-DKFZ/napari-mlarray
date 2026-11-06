"""napari-mlarray: geometry, layer types, boxes and round trips (no Qt needed)."""

import warnings

import numpy as np
import pytest
from mlarray import MLArray
from napari.components import ViewerModel

import napari_mlarray
from napari_mlarray import read_layers, write_single_image
from napari_mlarray._data import DisplayArray
from napari_mlarray._display import letters_to_lps

LPS_AXES = [
    {"name": "x", "unit": "millimeter", "increases_toward": "left"},
    {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
    {"name": "z", "unit": "millimeter", "increases_toward": "superior"},
]
RAS_AXES = [
    {"name": "x", "unit": "millimeter", "increases_toward": "right"},
    {"name": "y", "unit": "millimeter", "increases_toward": "anterior"},
    {"name": "z", "unit": "millimeter", "increases_toward": "superior"},
]


def _rot(deg_z, deg_x):
    a, b = np.deg2rad(deg_z), np.deg2rad(deg_x)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rz @ rx


# columns: array axes (z, y, x) of a (6, 7, 8) array; rows: space axes
GRIDS = {
    "lps": (LPS_AXES, np.array([[0, 0, 1.5], [0, 1.25, 0], [2.0, 0, 0]])),
    "ras": (RAS_AXES, np.array([[0, 0, 1.5], [0, 1.25, 0], [2.0, 0, 0]])),
    "sagittal": (LPS_AXES, np.array([[0, 0, -2.0], [1.25, 0, 0], [0, -1.5, 0]])),
    "mirrored": (LPS_AXES, np.array([[0, 0, -1.5], [0, 1.25, 0], [-2.0, 0, 0]])),
}


def _patient_file(path, linear, space_axes=LPS_AXES, shape=(6, 7, 8), dtype=np.int16, values=None,
                   profiles=None, origin=(10.0, -20.0, 30.0), axes=("z", "y", "x"), array=None):
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = origin
    array = np.arange(np.prod(shape), dtype=dtype).reshape(shape) if array is None else array
    space = {"name": "patient", "axes": space_axes,
             "from_array": {"array_axes": [a for a in axes if a in ("z", "y", "x")], "matrix": matrix.tolist()}}
    kwargs = {"values": values} if values else {}
    if profiles:
        kwargs["profiles"] = profiles
    MLArray(array, axes=list(axes) if all(isinstance(a, str) for a in axes) else axes, spaces=[space],
            **kwargs).save(path)
    return array, matrix


def _space_to_lps(space_axes):
    to_lps = np.zeros((3, 3))
    table = {"left": (0, 1), "right": (0, -1), "posterior": (1, 1), "anterior": (1, -1),
             "superior": (2, 1), "inferior": (2, -1)}
    for r, axis in enumerate(space_axes):
        lps_axis, sign = table[axis["increases_toward"]]
        to_lps[lps_axis, r] = sign
    return to_lps


def _add(path, **options):
    viewer = ViewerModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for data, kwargs, layer_type in read_layers(path, **options):
            viewer._add_layer_from_data(data, kwargs, layer_type)
    return viewer


@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("convention", ["radiological", "neurological"])
def test_every_voxel_is_at_its_world_position(tmp_path, grid, convention):
    space_axes, linear = GRIDS[grid]
    array, matrix = _patient_file(tmp_path / "a.mla", linear, space_axes)
    viewer = _add(tmp_path / "a.mla", convention=convention)
    layer = viewer.layers[0]
    letters = ("I", "P", "L") if convention == "radiological" else ("I", "P", "R")
    assert tuple(layer.axis_labels) == letters
    rotation = letters_to_lps(letters)
    # napari can slice it in 2-D: the affine has no permutation (data dim d runs along world dim d)
    block = np.asarray(layer.affine.affine_matrix)[:3, :3]
    assert np.allclose(block, np.diag(np.diag(block)))
    record = layer.metadata["napari_mlarray"]
    for index in np.ndindex(*array.shape):
        lps = _space_to_lps(space_axes) @ (matrix[:3, :3] @ np.array(index) + matrix[:3, 3])
        data_index = [index[a] for a in record["dims"]]
        np.testing.assert_allclose(layer.data_to_world(data_index), rotation @ lps, atol=1e-9)
        assert layer.data[tuple(data_index)] == array[index]


def test_world_is_right_handed_for_radiological_and_marks_neurological():
    from napari_mlarray._display import world_frame
    from mlarray.metadata import Space

    space = Space.from_dict({"name": "patient", "axes": LPS_AXES,
                             "from_array": {"array_axes": ["z", "y", "x"], "matrix": np.eye(4).tolist()}})
    assert world_frame(space, "radiological").camera_depth == "towards"
    assert world_frame(space, "neurological").camera_depth == "away"
    ras = Space.from_dict({"name": "patient", "axes": RAS_AXES,
                           "from_array": {"array_axes": ["z", "y", "x"], "matrix": np.eye(4).tolist()}})
    assert world_frame(ras, "radiological").labels == ["I", "P", "L"]


@pytest.mark.parametrize("oblique", ["keep", "nearest", "resample"])
def test_oblique_grids(tmp_path, oblique):
    linear = _rot(25, 10) @ np.array([[0, 0, 1.0], [0, 1.25, 0], [1.5, 0, 0]])
    array = np.zeros((20, 22, 24), np.int16)
    array[5:15, 6:16, 7:17] = 100
    _, matrix = _patient_file(tmp_path / "o.mla", linear, array=array, shape=array.shape)
    viewer = _add(tmp_path / "o.mla", oblique=oblique)
    layer = viewer.layers[0]
    record = layer.metadata["napari_mlarray"]
    centre = np.array([10.0, 11.0, 12.0])
    world = letters_to_lps(layer.axis_labels) @ (matrix[:3, :3] @ centre + matrix[:3, 3])
    block = np.asarray(layer.affine.affine_matrix)[:3, :3]
    if oblique == "keep":
        np.testing.assert_allclose(layer.data_to_world(centre[record["dims"]]), world, atol=1e-9)
    elif oblique == "resample":
        assert record["resampled"]
        assert np.allclose(block, np.diag(np.diag(block)))
        data_index = np.round(layer.world_to_data(world)).astype(int)
        assert layer.data[tuple(data_index)] == 100
    else:
        assert np.allclose(block, np.diag(np.diag(block)))


def test_labels_with_table_names_colours_and_codes(tmp_path):
    labels = {"0": {"name": "background"}, "1": {"name": "liver", "color": [221, 130, 101],
                                                  "code": {"scheme": "SCT", "value": "10200004", "meaning": "Liver"}},
              "2": {"name": "tumor"}}
    array = np.zeros((4, 5, 6), np.uint8)
    array[1, 2, 3] = 1
    array[2, 2, 2] = 2
    _patient_file(tmp_path / "s.mla", GRIDS["lps"][1], array=array, shape=array.shape,
                  values={"kind": "label", "labels": labels})
    layer = _add(tmp_path / "s.mla").layers[0]
    assert layer._type_string == "labels"
    assert list(layer.features["name"]) == ["background", "liver", "tumor"]
    assert list(layer.features["code"])[1] == "Liver"
    np.testing.assert_allclose(layer.colormap.map(np.array([1]))[0][:3], np.array([221, 130, 101]) / 255, atol=1e-6)
    assert layer.colormap.map(np.array([0]))[0][3] == 0


def test_scaled_values_and_round_trip(tmp_path):
    array = (np.arange(6 * 7 * 8).reshape(6, 7, 8) % 50).astype(np.int16)
    _patient_file(tmp_path / "ct.mla", GRIDS["ras"][1], RAS_AXES, array=array,
                  values={"kind": "measurement", "scale": 2.0, "offset": -1024.0, "unit": "hounsfield_unit"})
    layer = _add(tmp_path / "ct.mla").layers[0]
    record = layer.metadata["napari_mlarray"]
    shown = np.asarray(layer.data)
    np.testing.assert_allclose(shown, np.transpose(array, record["dims"]) * 2.0 - 1024.0)
    data, state, _ = layer.as_layer_data_tuple()
    write_single_image(str(tmp_path / "back.mla"), data, state)
    back = MLArray(tmp_path / "back.mla")
    original = MLArray(tmp_path / "ct.mla")
    assert back.dtype == np.int16
    np.testing.assert_array_equal(back.to_numpy(), array)
    assert back.meta.values.scale == 2.0 and back.meta.values.offset == -1024.0
    np.testing.assert_allclose(back.spaces[0].matrix(), original.spaces[0].matrix(), atol=1e-12)
    assert back.spaces[0].axes == original.spaces[0].axes
    assert back.meta.history[-1]["action"] == "edited"


@pytest.mark.parametrize("grid", ["sagittal", "mirrored"])
def test_labels_round_trip_keeps_geometry_and_table(tmp_path, grid):
    space_axes, linear = GRIDS[grid]
    array = np.random.default_rng(0).integers(0, 3, (6, 7, 8)).astype(np.uint8)
    _patient_file(tmp_path / "seg.mla", linear, space_axes, array=array,
                  values={"kind": "label", "labels": {"1": {"name": "a"}, "2": {"name": "b"}}})
    layer = _add(tmp_path / "seg.mla").layers[0]
    layer.data[0, 0, 0] = 2            # paint
    data, state, _ = layer.as_layer_data_tuple()
    write_single_image(str(tmp_path / "out.mla"), data, state)
    back = MLArray(tmp_path / "out.mla")
    expected = array.copy()
    expected[0, 0, 0] = 2              # data index (0, 0, 0) is array index (0, 0, 0) in any axis order
    np.testing.assert_array_equal(back.to_numpy(), expected)
    np.testing.assert_allclose(back.spaces[0].matrix(), MLArray(tmp_path / "seg.mla").spaces[0].matrix(), atol=1e-12)
    assert back.meta.values.labels["2"]["name"] == "b"


def test_moved_layer_is_saved_where_it_is_shown(tmp_path):
    _patient_file(tmp_path / "a.mla", GRIDS["sagittal"][1])
    layer = _add(tmp_path / "a.mla").layers[0]
    layer.translate = [5.0, -3.0, 2.0]
    data, state, _ = layer.as_layer_data_tuple()
    write_single_image(str(tmp_path / "moved.mla"), data, state)
    m = MLArray(tmp_path / "moved.mla").spaces[0].matrix()
    rotation = letters_to_lps(layer.axis_labels)
    dims = layer.metadata["napari_mlarray"]["dims"]
    for index in [(0, 0, 0), (5, 6, 7), (2, 3, 1)]:
        data_index = [index[a] for a in dims]
        np.testing.assert_allclose(m[:3, :3] @ np.array(index) + m[:3, 3],
                                   rotation.T @ layer.data_to_world(data_index), atol=1e-9)


def test_new_labels_layer_is_written_in_the_patient_frame(tmp_path, monkeypatch):
    _patient_file(tmp_path / "a.mla", GRIDS["sagittal"][1])
    viewer = _add(tmp_path / "a.mla")
    image = viewer.layers[0]
    painted = np.zeros(image.data.shape, np.uint8)
    painted[1, 2, 3] = 1
    labels = viewer.add_labels(painted, affine=image.affine, features={"index": [1], "name": ["lesion"]})
    import napari

    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    data, state, _ = labels.as_layer_data_tuple()
    write_single_image(str(tmp_path / "new.mla"), data, state)
    new = MLArray(tmp_path / "new.mla")
    assert new.meta.values.kind == "label" and new.meta.values.labels["1"]["name"] == "lesion"
    assert [a.increases_toward for a in new.spaces[0].axes] == ["left", "posterior", "superior"]
    # the painted voxel is at the same LPS position as the image voxel under it
    rotation = letters_to_lps(image.axis_labels)
    world = image.data_to_world([1, 2, 3])
    m = new.spaces[0].matrix()
    np.testing.assert_allclose(m[:3, :3] @ np.array([1, 2, 3]) + m[:3, 3], rotation.T @ world, atol=1e-9)


def test_channels_become_one_layer_each(tmp_path):
    array = np.random.default_rng(1).random((3, 4, 5, 6)).astype(np.float32)
    axes = [{"name": "c", "type": "channel", "labels": ["T1", "T2", "FLAIR"]}, "z", "y", "x"]
    _patient_file(tmp_path / "mc.mla", GRIDS["lps"][1], array=array, shape=array.shape, axes=axes)
    viewer = _add(tmp_path / "mc.mla")
    assert [layer.name for layer in viewer.layers] == ["mc [T1]", "mc [T2]", "mc [FLAIR]"]
    flair = viewer.layers[2]
    assert flair.ndim == 3 and tuple(flair.axis_labels) == ("I", "P", "L")
    data, state, _ = flair.as_layer_data_tuple()
    write_single_image(str(tmp_path / "flair.mla"), data, state)
    back = MLArray(tmp_path / "flair.mla")
    assert back.shape == (1, 4, 5, 6) and list(back.axes[0].labels) == ["FLAIR"]
    np.testing.assert_allclose(back.to_numpy()[0], array[2])


def test_time_axis_is_scaled(tmp_path):
    array = np.zeros((4, 3, 5, 6), np.float32)
    axes = [{"name": "t", "type": "time", "start": 0.0, "step": 2.5, "unit": "second"}, "z", "y", "x"]
    _patient_file(tmp_path / "t.mla", GRIDS["lps"][1], array=array, shape=array.shape, axes=axes)
    layer = _add(tmp_path / "t.mla").layers[0]
    assert tuple(layer.axis_labels) == ("t", "I", "P", "L")
    assert layer.data_to_world([2, 0, 0, 0])[0] == 5.0


def test_single_slice_in_a_patient_space(tmp_path):
    matrix = np.array([[0.8, 0, -100.0], [0, 0.8, -100.0], [0, 0, 42.0], [0, 0, 1]])
    space = {"name": "patient", "axes": LPS_AXES, "from_array": {"array_axes": ["x", "y"], "matrix": matrix.tolist()}}
    array = np.arange(20, dtype=np.int16).reshape(4, 5)
    MLArray(array, axes=["y", "x"], spaces=[space]).save(tmp_path / "slice.mla")
    layer = _add(tmp_path / "slice.mla").layers[0]
    assert layer.ndim == 3
    record = layer.metadata["napari_mlarray"]
    data_index = [0 if a is None else {0: 2, 1: 3}[a] for a in record["dims"]]
    lps = matrix[:3, :2] @ np.array([3, 2]) + matrix[:3, 2]
    np.testing.assert_allclose(layer.data_to_world(data_index), letters_to_lps(layer.axis_labels) @ lps, atol=1e-9)
    data, state, _ = layer.as_layer_data_tuple()
    write_single_image(str(tmp_path / "slice_back.mla"), data, state)
    back = MLArray(tmp_path / "slice_back.mla")
    np.testing.assert_array_equal(back.to_numpy(), array)
    np.testing.assert_allclose(back.spaces[0].matrix(), matrix, atol=1e-9)


def test_index_space_image(tmp_path):
    MLArray(np.arange(12, dtype=np.uint8).reshape(3, 4), axes=["y", "x"]).save(tmp_path / "photo.mla")
    layer = _add(tmp_path / "photo.mla").layers[0]
    np.testing.assert_array_equal(np.asarray(layer.affine.affine_matrix), np.eye(3))
    assert tuple(layer.axis_labels) == ("y", "x")


def test_boxes_3d_are_shells_aligned_with_the_image(tmp_path):
    boxes = {"frame": "index", "axes": ["z", "y", "x"],
             "items": [{"min": [1, 2, 3], "max": [4, 5, 6], "label": 1, "score": 0.9}]}
    _patient_file(tmp_path / "b.mla", GRIDS["sagittal"][1], shape=(6, 7, 8),
                  values={"kind": "measurement"}, profiles={"ml": {"boxes": boxes}})
    viewer = _add(tmp_path / "b.mla")
    image, shells, texts = viewer.layers
    assert shells._type_string == "labels" and shells.features["score"][0] == 0.9
    assert texts._type_string == "points" and texts.features["box"][0] == "1 0.900"
    full = np.asarray(shells.data)
    on = np.argwhere(full > 0)
    world = np.array([shells.data_to_world(i) for i in on])
    rotation = letters_to_lps(image.axis_labels)
    m = MLArray(tmp_path / "b.mla").spaces[0].matrix()
    corners = np.array([[z, y, x] for z in (1, 4) for y in (2, 5) for x in (3, 6)], float)
    corner_world = (rotation @ (m[:3, :3] @ corners.T + m[:3, 3:4])).T
    assert np.allclose(world.min(0), corner_world.min(0)) and np.allclose(world.max(0), corner_world.max(0))
    # 2-D slices show where they cut the box surface (the full volume, napari's 3-D view, is the wireframe)
    from napari_bboxes import BoxShells

    faces = BoxShells(shells.data.shape, shells.data.bounds, mode="faces")
    for d in range(3):
        for k in range(full.shape[d]):
            key = (slice(None),) * d + (k,)
            assert np.array_equal(np.asarray(shells.data[key]), faces[key])
    # the box interior is empty (a shell, not a filled box)
    lo, hi = on.min(0), on.max(0)
    assert not full[tuple(slice(a + 1, b) for a, b in zip(lo, hi))].any()


def test_boxes_2d_are_rectangles(tmp_path):
    boxes = {"frame": "index", "axes": ["y", "x"], "items": [{"min": [1, 2], "max": [3, 5], "label": "lesion"}]}
    MLArray(np.zeros((8, 9), np.uint8), axes=["y", "x"], profiles={"ml": {"boxes": boxes}}).save(tmp_path / "b2.mla")
    viewer = _add(tmp_path / "b2.mla")
    shapes = viewer.layers[1]
    assert shapes._type_string == "shapes"
    np.testing.assert_allclose(shapes.data[0], [[1, 2], [1, 5], [3, 5], [3, 2]])


def test_box_only_file(tmp_path):
    boxes = {"frame": "index", "axes": ["z", "y", "x"], "items": [{"min": [1, 1, 1], "max": [3, 4, 5]}]}
    MLArray(axes=["z", "y", "x"], profiles={"ml": {"boxes": boxes}}).save(tmp_path / "only.mla")
    viewer = _add(tmp_path / "only.mla")
    assert len(viewer.layers) == 1 and viewer.layers[0].data.shape == (4, 5, 6)


def test_lazy_view_matches_numpy():
    source = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    view = DisplayArray(source, [3, None, 1, 0, 2], scale=2.0, offset=1.0, dtype=np.float32)
    expected = np.expand_dims(np.transpose(source, [3, 1, 0, 2]), 1) * 2.0 + 1.0
    assert view.shape == expected.shape
    for key in [(slice(None),), (1,), (2, 0, slice(1, 3)), (Ellipsis, 3), (slice(None), slice(None), 2, 1, slice(None))]:
        np.testing.assert_allclose(view[key], expected[key])


def test_large_files_are_read_lazily(tmp_path, monkeypatch):
    _patient_file(tmp_path / "big.mla", GRIDS["sagittal"][1])
    monkeypatch.setattr("napari_mlarray._reader.LAZY_BYTES", 0)
    layer = _add(tmp_path / "big.mla").layers[0]
    assert isinstance(layer.data, DisplayArray)
    eager = MLArray(tmp_path / "big.mla").to_numpy()
    np.testing.assert_array_equal(np.asarray(layer.data), np.transpose(eager, layer.metadata["napari_mlarray"]["dims"]))


def test_reader_accepts_only_mla():
    assert callable(napari_mlarray.napari_get_reader("a.mla"))
    assert napari_mlarray.napari_get_reader(["a.mla", "b.mla"]) is not None
    assert napari_mlarray.napari_get_reader("a.nii.gz") is None


def test_negligible_rotation_is_snapped_not_resampled_and_written_back_exactly(tmp_path):
    linear = _rot(0.0014, 0.0) @ GRIDS["lps"][1]          # scanner round-off, far below a voxel
    array = np.random.default_rng(3).integers(0, 3, (20, 30, 40)).astype(np.uint8)
    _, matrix = _patient_file(tmp_path / "r.mla", linear, array=array, shape=array.shape,
                              values={"kind": "label"})
    layer = _add(tmp_path / "r.mla").layers[0]
    record = layer.metadata["napari_mlarray"]
    assert not record["resampled"] and layer.data.shape == array.shape
    block = np.asarray(layer.affine.affine_matrix)[:3, :3]
    assert np.allclose(block, np.diag(np.diag(block)))
    data, state, _ = layer.as_layer_data_tuple()
    write_single_image(str(tmp_path / "back.mla"), data, state)
    back = MLArray(tmp_path / "back.mla")
    np.testing.assert_allclose(back.spaces[0].matrix(), matrix, atol=1e-12)
    np.testing.assert_array_equal(back.to_numpy(), array)


def test_new_layer_in_a_viewer_without_patient_labels(tmp_path, monkeypatch):
    import napari

    viewer = ViewerModel()
    layer = viewer.add_labels(np.zeros((4, 5, 6), np.uint8), scale=(2.0, 1.0, 0.5), translate=(1.0, 2.0, 3.0))
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    data, state, _ = layer.as_layer_data_tuple()
    write_single_image(str(tmp_path / "plain.mla"), data, state)
    space = MLArray(tmp_path / "plain.mla").spaces[0]
    assert space.name == "world" and [a.name for a in space.axes] == ["z", "y", "x"]
    np.testing.assert_allclose(space.matrix()[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(np.diag(space.matrix())[:3], [2, 1, 0.5])


def test_patient_world_is_found_from_the_layers_when_dims_have_no_labels(tmp_path, monkeypatch):
    import napari

    _patient_file(tmp_path / "a.mla", GRIDS["sagittal"][1])
    viewer = _add(tmp_path / "a.mla")
    viewer.dims.axis_labels = ("-3", "-2", "-1")      # as napari < 0.8 leaves them
    image = viewer.layers[0]
    labels = viewer.add_labels(np.zeros(image.data.shape, np.uint8), affine=image.affine)
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    data, state, _ = labels.as_layer_data_tuple()
    write_single_image(str(tmp_path / "new.mla"), data, state)
    assert [a.increases_toward for a in MLArray(tmp_path / "new.mla").spaces[0].axes] == ["left", "posterior", "superior"]
