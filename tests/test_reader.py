import os
from pathlib import Path

import numpy as np

from mlarray import MLArray, Meta

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp")

from napari.layers import Shapes, Vectors
from napari_mlarray._reader import (
    _BBOX3D_EDGE_VERTEX_INDICES,
    bboxes_minmax_to_napari_rectangles_2d,
    bboxes_minmax_to_napari_vectors_3d,
    napari_get_reader,
    reader_function,
)


def test_get_reader_accepts_mlarray_suffix():
    assert callable(napari_get_reader("sample.mla"))
    assert napari_get_reader("sample.npy") is None


def test_bboxes_minmax_to_napari_rectangles_2d_preserves_2d_behavior():
    bboxes = np.array([[[10, 30], [20, 40]]], dtype=np.float32)

    rectangles = bboxes_minmax_to_napari_rectangles_2d(bboxes)

    expected = np.array(
        [[[10, 30], [10, 40], [20, 40], [20, 30]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(rectangles, expected)


def test_bboxes_minmax_to_napari_vectors_3d_builds_wireframe_edges():
    bboxes = np.array([[[1, 4], [2, 5], [3, 6]]], dtype=np.float32)

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
    np.testing.assert_allclose(
        data,
        np.array([[[10, 30], [10, 40], [20, 40], [20, 30]]], dtype=np.float32),
    )


def test_reader_function_returns_vectors_layer_for_3d_bboxes(tmp_path):
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
    vectors_data, vectors_kwargs, vectors_layer_type = layer_data[1]

    assert image_layer_type == "image"
    assert vectors_layer_type == "vectors"
    assert vectors_data.shape == (24, 2, 3)
    assert vectors_kwargs["vector_style"] == "line"
    assert vectors_kwargs["edge_width"] == 2
    assert vectors_kwargs["opacity"] == 1.0
    assert vectors_kwargs["edge_color"].shape == (24, 4)
    assert set(vectors_kwargs["features"]) == {"box_index", "score", "label"}
    np.testing.assert_array_equal(
        vectors_kwargs["features"]["box_index"],
        np.repeat(np.array([0, 1], dtype=np.int32), 12),
    )
    np.testing.assert_allclose(
        vectors_kwargs["features"]["score"],
        np.repeat(np.array([0.9, 0.7], dtype=np.float32), 12),
    )
    np.testing.assert_array_equal(
        vectors_kwargs["features"]["label"],
        np.repeat(np.array(["tumor", "node"], dtype=object), 12),
    )
    layer = Vectors(vectors_data, **vectors_kwargs)
    assert str(layer.vector_style) == "line"
    np.testing.assert_allclose(
        vectors_kwargs["affine"],
        np.array(
            [
                [-3.0, 0.0, 0.0, 45.0],
                [0.0, -2.0, 0.0, 28.0],
                [0.0, 0.0, -1.5, 14.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
