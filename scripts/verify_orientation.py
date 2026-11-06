"""On-screen check of napari-mlarray's display convention (needs a display or Xvfb: ``xvfb-run python ...``).

A letter phantom is defined once in patient (LPS) millimetres and stored on several grids: LPS, RAS, sagittal
storage order and a 25 degree oblique grid. Each is opened in napari and screenshotted in the axial, coronal and
sagittal planes; the script then locates the bright letter and the brighter (patient-left) half on screen.

Radiological ground truth (every file must give the same picture):
    axial:    'F' upper right, upright; brighter half right   (anterior up, patient left on the right)
    coronal:  'F' upper LEFT (patient right, superior); brighter half right
    sagittal: 'F' lower right (posterior, inferior)
Neurological mirrors axial and coronal only. 3-D: an attenuated MIP with a dim plate superior and the bright
'F' inferior must show the F in front (radiological) or the plate in front (neurological): otherwise the 3-D view
is mirrored.

Usage: python scripts/verify_orientation.py OUTDIR
"""

import sys
import warnings
from pathlib import Path

import numpy as np

BODY = np.array([58.0, 70.0, 58.0])     # ellipsoid semi-axes (x, y, z), mm
LPS_AXES = [{"name": "x", "unit": "millimeter", "increases_toward": "left"},
            {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
            {"name": "z", "unit": "millimeter", "increases_toward": "superior"}]


def letter_f(u, v, u0, v0, w=30.0, h=40.0, t=8.0):
    uu, vv = u - u0, v - v0
    inside = (uu >= 0) & (uu <= w) & (vv >= 0) & (vv <= h)
    return inside & ((uu <= t) | (vv >= h - t) | ((vv >= h / 2 - t / 2) & (vv <= h / 2 + t / 2) & (uu <= 0.75 * w)))


def content(x, y, z):
    """Intensity at LPS coordinates."""
    body = (x / BODY[0]) ** 2 + (y / BODY[1]) ** 2 + (z / BODY[2]) ** 2 <= 1
    value = np.where(body, 60.0 + 40.0 * (x > 0), 0.0)
    letters = letter_f(x, -y, 8.0, 8.0) | letter_f(x, z, -45.0, 8.0) | letter_f(y, z, 8.0, -48.0)
    return np.where(body & letters, 255.0, value)


def rotation(axis, degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def write_phantom(path, direction, spacing, extent):
    """Store the phantom on a grid with ITK-style direction (columns: file axes i, j, k) as array [k, j, i]."""
    from mlarray import MLArray

    direction, spacing = np.asarray(direction, float), np.asarray(spacing, float)
    size = np.round(np.asarray(extent) / spacing).astype(int)
    origin = -direction @ (spacing * (size - 1) / 2.0)
    i, j, k = np.meshgrid(*[np.arange(n) for n in size], indexing="ij")
    xyz = (direction * spacing) @ np.stack([i, j, k]).reshape(3, -1) + origin[:, None]
    array = content(*xyz).reshape(size).transpose(2, 1, 0).astype(np.int16)   # [k, j, i]
    matrix = np.eye(4)
    matrix[:3, :3] = (direction * spacing)[:, ::-1]                          # columns k, j, i = array axes z, y, x
    matrix[:3, 3] = origin
    MLArray(np.ascontiguousarray(array), axes=["z", "y", "x"], spaces=[{
        "name": "patient", "axes": LPS_AXES, "from_array": {"array_axes": ["z", "y", "x"], "matrix": matrix.tolist()}
    }]).save(path)


GRIDS = {
    "lps": (np.eye(3), (1.0, 1.25, 1.5), (150.0, 160.0, 140.0)),
    "ras": (np.diag([-1.0, -1.0, 1.0]), (1.0, 1.25, 1.5), (150.0, 160.0, 140.0)),
    "sagittal": (np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]), (1.25, 1.5, 1.0), (160.0, 140.0, 150.0)),
    "oblique": (rotation("z", 25) @ rotation("x", 10), (1.0, 1.25, 1.5), (150.0, 160.0, 140.0)),
}
PLANES = {"axial": (0, 1, 2), "coronal": (1, 0, 2), "sagittal": (2, 0, 1)}
EXPECTED = {
    "radiological": {"axial": "F (right, upper); brighter right", "coronal": "F (left, upper); brighter right",
                     "sagittal": "F (right, lower); brighter right"},
    "neurological": {"axial": "F (left, upper); brighter left", "coronal": "F (right, upper); brighter left",
                     "sagittal": "F (right, lower); brighter right"},
}


def describe(image):
    g = image.mean(-1)
    ys, xs = np.nonzero(g > 240)
    body = g > 30
    by, bx = np.nonzero(body)
    if len(bx) == 0 or len(xs) == 0:
        return "nothing visible"
    cx = bx.mean()
    left = g[:, : int(cx)][body[:, : int(cx)]].mean()
    right = g[:, int(cx):][body[:, int(cx):]].mean()
    return (f"F ({'right' if xs.mean() > cx else 'left'}, {'lower' if ys.mean() > by.mean() else 'upper'}); "
            f"brighter {'right' if right > left else 'left'}")


def main(out):
    import imageio.v3 as iio
    import napari

    import napari_mlarray
    from napari_mlarray._display import camera

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, grid in GRIDS.items():
        write_phantom(out / f"{name}.mla", *grid)
    viewer = napari.Viewer(show=True)
    viewer.window.resize(900, 900)

    def shot(name):
        viewer.reset_view()
        image = viewer.screenshot(canvas_only=True, flash=False)[..., :3]
        iio.imwrite(out / f"{name}.png", image)
        return image

    failures = 0
    print(f"napari {napari.__version__}, napari-mlarray {napari_mlarray.__version__}")
    for convention in ("radiological", "neurological"):
        for name in GRIDS:
            viewer.layers.clear()
            viewer.dims.ndisplay = 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                layer = napari_mlarray.add_mlarray(viewer, out / f"{name}.mla", convention=convention)[0]
            layer.contrast_limits = (0, 255)
            for plane, order in PLANES.items():
                viewer.dims.order = order
                viewer.dims.current_step = tuple(s // 2 for s in viewer.dims.nsteps)
                seen = describe(shot(f"{convention}_{name}_{plane}"))
                ok = seen == EXPECTED[convention][plane]
                failures += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {convention:12s} {name:9s} {plane:9s} {seen}")

    # 3-D handedness
    n = 64
    x = np.arange(n) - (n - 1) / 2.0
    z, y, x3 = np.meshgrid(x, x, x, indexing="ij")
    volume = np.zeros((n, n, n), np.float32)
    volume[(z > 18) & (z < 26)] = 60
    volume[letter_f(x3, -y, -15.0, -20.0) & (z < -15) & (z > -25)] = 255
    matrix = np.zeros((4, 4))
    matrix[0, 2] = matrix[1, 1] = matrix[2, 0] = 1
    matrix[:3, 3] = x[0]
    matrix[3, 3] = 1
    from mlarray import MLArray

    MLArray(volume, axes=["z", "y", "x"], spaces=[{"name": "patient", "axes": LPS_AXES, "from_array": {
        "array_axes": ["z", "y", "x"], "matrix": matrix.tolist()}}]).save(out / "chirality.mla")
    for convention, expect in (("radiological", "F"), ("neurological", "plate")):
        viewer.layers.clear()
        camera(viewer).orientation = ("towards", "down", "right")
        viewer.dims.ndisplay = 3
        layer = napari_mlarray.add_mlarray(viewer, out / "chirality.mla", convention=convention)[0]
        layer.rendering, layer.attenuation, layer.contrast_limits = "attenuated_mip", 0.5, (0, 255)
        orientation = tuple(o.value for o in camera(viewer).orientation)
        front = "F" if (shot(f"3d_{convention}").mean(-1) > 200).mean() > 0.005 else "plate"
        failures += front != expect
        print(f"{'ok  ' if front == expect else 'FAIL'} {convention:12s} 3-D       {front} in front (expected {expect}), camera {orientation}")
    viewer.close()
    print("all checks passed" if not failures else f"{failures} checks failed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
