import os
from pathlib import Path

import numpy as np

from mlarray import MLArray, Meta

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp")

from napari.layers import Shapes, Surface
from napari_mlarray._reader import (
    _BBOX3D_EDGE_VERTEX_INDICES,
    _BBOX3D_FACE_TRIANGLE_VERTEX_INDICES,
    bboxes_minmax_to_napari_rectangles_2d,
    bboxes_minmax_to_napari_surface_3d,
    bboxes_minmax_to_napari_vectors_3d,
    napari_get_reader,
    reader_function,
)


def test_get_reader_accepts_mlarray_suffix():
    assert callable(napari_get_reader("sample.mla"))
    assert napari_get_reader("sample.npy") is None


def test_bboxes_minmax_to_napari_rectangles_2d_preserves_2d_behavior():
    # Input is (N, D=2, 2) in SAR+ ZYX order: dim0=[10,20], dim1=[30,40]
    bboxes = np.array([[[10, 20], [30, 40]]], dtype=np.float32)

    rectangles = bboxes_minmax_to_napari_rectangles_2d(bboxes)

    expected = np.array(
        [[[10, 30], [10, 40], [20, 40], [20, 30]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(rectangles, expected)


def test_bboxes_minmax_to_napari_vectors_3d_builds_wireframe_edges():
    # Input is (N, D=3, 2) in SAR+ ZYX order: Z=[3,6], Y=[2,5], X=[1,4]
    bboxes = np.array([[[3, 6], [2, 5], [1, 4]]], dtype=np.float32)

    vectors = bboxes_minmax_to_napari_vectors_3d(bboxes)

    assert vectors.shape == (12, 2, 3)

    starts = vectors[:, 0]
    ends = starts + vectors[:, 1]
    corners = {
        (3.0, 2.0, 1.0),
        (3.0, 2.0, 4.0),
        (3.0, 5.0, 1.0),
        (3.0, 5.0, 4.0),
        (6.0, 2.0, 1.0),
        (6.0, 2.0, 4.0),
        (6.0, 5.0, 1.0),
        (6.0, 5.0, 4.0),
    }
    observed_edges = set()
    for start, end, delta in zip(starts, ends, vectors[:, 1], strict=False):
        assert tuple(start.tolist()) in corners
        assert tuple(end.tolist()) in corners
        assert np.count_nonzero(delta) == 1
        observed_edges.add(
            tuple(sorted((tuple(start.tolist()), tuple(end.tolist()))))
        )

    assert len(observed_edges) == len(_BBOX3D_EDGE_VERTEX_INDICES)


def test_bboxes_minmax_to_napari_surface_3d_builds_cuboid_mesh():
    # Input is (N, D=3, 2) in SAR+ ZYX order: Z=[3,6], Y=[2,5], X=[1,4]
    bboxes = np.array([[[3, 6], [2, 5], [1, 4]]], dtype=np.float32)

    vertices, faces, values = bboxes_minmax_to_napari_surface_3d(bboxes)

    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    assert values.shape == (8,)
    assert np.all(values == 0)
    corners = {
        (3.0, 2.0, 1.0),
        (3.0, 2.0, 4.0),
        (3.0, 5.0, 1.0),
        (3.0, 5.0, 4.0),
        (6.0, 2.0, 1.0),
        (6.0, 2.0, 4.0),
        (6.0, 5.0, 1.0),
        (6.0, 5.0, 4.0),
    }
    assert {tuple(vertex.tolist()) for vertex in vertices} == corners
    assert np.all(faces >= 0)
    assert np.all(faces < len(vertices))
    assert len(faces) == len(_BBOX3D_FACE_TRIANGLE_VERTEX_INDICES)


def test_reader_function_returns_shapes_layer_for_2d_bbox_only(tmp_path):
    path = Path(tmp_path) / "bbox2d.mla"
    MLArray(
        meta=Meta(bbox={"bboxes": [[[10, 30], [20, 40]]], "scores": [0.9], "labels": ["lesion"]})
    ).save(path)

    layer_data = reader_function(str(path))

    assert len(layer_data) == 1
    data, kwargs, layer_type = layer_data[0]
    assert layer_type == "shapes"
    assert kwargs["shape_type"] == "rectangle"
    assert kwargs["face_color"] == "transparent"
    assert "text" in kwargs
    layer = Shapes(data, **kwargs)
    assert layer.shape_type[0] == "rectangle"
    # MLArray stored dim0=[10,30], dim1=[20,40] in (N,D,2) format.
    # SAR+ reversal (no affine): new dim0=[20,40], new dim1=[10,30].
    # Rectangle corners: [20,10], [20,30], [40,30], [40,10].
    np.testing.assert_allclose(
        data,
        np.array([[[20, 10], [20, 30], [40, 30], [40, 10]]], dtype=np.float32),
    )


def test_reader_function_returns_surface_layer_for_3d_bboxes(tmp_path):
    path = Path(tmp_path) / "bbox3d.mla"
    array = np.zeros((4, 5, 6), dtype=np.float32)
    image = MLArray(
        array,
        spacing=(1.5, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
        meta=Meta(
            bbox={
                "bboxes": [
                    [[1, 4], [2, 5], [3, 6]],
                    [[0, 1], [1, 3], [2, 4]],
                ],
                "scores": [0.9, 0.7],
                "labels": ["tumor", "node"],
            }
        ),
    )
    image.save(path)

    layer_data = reader_function(str(path))

    assert len(layer_data) == 2
    _, _, image_layer_type = layer_data[0]
    surface_data, surface_kwargs, surface_layer_type = layer_data[1]

    assert image_layer_type == "image"
    assert surface_layer_type == "surface"
    assert surface_kwargs["blending"] == "translucent"
    assert surface_kwargs["opacity"] == 1.0
    assert surface_kwargs["shading"] == "flat"
    assert surface_kwargs["vertex_colors"].shape == (16, 4)
    vertices, faces, values = surface_data
    assert vertices.shape == (16, 3)
    assert faces.shape == (24, 3)
    assert values.shape == (16,)
    np.testing.assert_array_equal(
        values,
        np.repeat(np.array([0, 1], dtype=np.float32), 8),
    )
    surface_layer = Surface(surface_data, **surface_kwargs)
    assert str(surface_layer.shading) == "flat"
    # scale / translate follow the same SAR+ negative-scale convention as the image layer.
    # array (4,5,6) XYZ → SAR+ ZYX (6,5,4); spacing=(1.5,2,3) → SAR sp=[3,2,1.5];
    # origin=(10,20,30) → SAR orig=[30,20,10].
    np.testing.assert_allclose(surface_kwargs["scale"], [-3.0, -2.0, -1.5])
    np.testing.assert_allclose(surface_kwargs["translate"], [45.0, 28.0, 14.5])
