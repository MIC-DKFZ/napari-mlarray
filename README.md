# napari-mlarray

<p align="center">
  <a href="https://pypi.org/project/napari-mlarray/"><img src="https://img.shields.io/pypi/v/napari-mlarray?logo=pypi&color=brightgreen" alt="PyPI" align="middle" /></a>
  <a href="https://pypi.org/project/napari-mlarray/"><img src="https://img.shields.io/pypi/pyversions/napari-mlarray?logo=python" alt="Python Version" align="middle" /></a>
  <a href="https://github.com/MIC-DKFZ/napari-mlarray/actions"><img src="https://img.shields.io/github/actions/workflow/status/MIC-DKFZ/napari-mlarray/test_and_deploy.yml?branch=main&logo=github" alt="Tests" align="middle" /></a>
  <a href="https://napari-hub.org/plugins/napari-mlarray"><img src="https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-mlarray" alt="napari hub" align="middle" /></a>
  <a href="https://github.com/MIC-DKFZ/napari-mlarray/blob/main/LICENSE"><img src="https://img.shields.io/github/license/MIC-DKFZ/napari-mlarray" alt="License" align="middle" /></a>
</p>

A [napari](https://napari.org) reader and writer for [MLArray](https://github.com/MIC-DKFZ/mlarray) (`.mla`) files
that shows every image where it physically is, in the orientation radiologists expect.

```bash
pip install napari-mlarray            # into an existing napari installation
pip install "napari-mlarray[all]"     # with napari and Qt
```

Drag a `.mla` file into napari, or `File > Open`. To convert NIfTI, NRRD, MetaImage, MGH or DICOM to `.mla`, use
[MedVol](https://mic-dkfz.github.io/MedVol/): `medvol convert ct.nii.gz ct.mla`.

## What you see

napari knows nothing about patient anatomy: a layer is an array plus an affine into "world" coordinates, and the
2-D canvas shows the last two world dims (down and right). napari-mlarray chooses the world so that the default
camera gives the familiar picture, with no flips or rotations to fix by hand:

| plane (`viewer.dims.order`) | screen up | screen right |
|---|---|---|
| axial `(0, 1, 2)` | anterior | patient left (radiological) |
| coronal `(1, 0, 2)` | superior | patient left (radiological) |
| sagittal `(2, 0, 1)` | superior | posterior |

- The world axes of a patient image point to **I, P, L** (labelled in napari's dims slider). This frame is
  right-handed, so the 3-D view is not mirrored. With the `neurological` convention they point to **I, P, R**
  (patient left on the screen's left) and the plugin flips the 3-D camera depth to keep 3-D unmirrored.
- Every voxel is placed exactly from the file's geometry (`spaces[0]`): any storage order (axial, sagittal,
  coronal, flipped) gives the same picture, and images of different grids overlay correctly. The array is
  reordered for display only, never resampled, unless the grid is oblique (below).
- World coordinates are millimetres. Convert the cursor position back to the file's patient coordinates (DICOM
  LPS) with `napari_mlarray.letters_to_lps(viewer.dims.axis_labels[-3:]).T @ viewer.cursor.position[-3:]`.

This is verified on screen by [`scripts/verify_orientation.py`](scripts/verify_orientation.py) (napari 0.10): a
letter phantom stored axially, RAS-flipped, sagittally and obliquely gives the same, correct picture in all three
planes and an unmirrored 3-D view, in both conventions. Other spaces work the same way:
earth-observation spaces are shown north up and east right, spaces without direction words in reversed axis order
(`z, y, x`), so an identity matrix shows the array as NumPy does.

### Oblique images

napari cannot slice an affine that rotates the displayed axes against the slicing axis in 2-D (it drops the
out-of-plane rotation). For grids that are oblique to the world axes, choose:

| `oblique` | effect |
|---|---|
| `resample` (default) | resampled onto a world-aligned grid (linear for images, nearest neighbour for labels): exact positions |
| `nearest` | axes snapped to the closest world axes, as ITK-SNAP and MRIcroGL do: upright, but positions are approximate |
| `keep` | the exact affine: correct in 3-D, approximate in 2-D slices |

### Settings

```python
import napari_mlarray

napari_mlarray.set_display(convention="neurological", oblique="nearest")   # for files opened from now on
layers = napari_mlarray.add_mlarray(viewer, "ct.mla", convention="radiological")   # one file, explicit options
```

or set `NAPARI_MLARRAY_CONVENTION` / `NAPARI_MLARRAY_OBLIQUE` before starting napari.

## Layers

| content of the file | napari layer |
|---|---|
| `values.kind` `measurement` (and others) | Image; `values.scale`/`offset` applied (e.g. Hounsfield units), contrast limits from `ml` statistics when present |
| `values.kind` `label` or `instance` | Labels, with names, colours and codes of `values.labels` (shown when hovering) |
| a `channel` axis | one Image layer per channel, named after the axis labels (`ct [T1]`, `ct [T2]`, ...) |
| `time` and other axes | sliders; time is in its unit (`start`, `step`) |
| a 2-D grid in a 3-D space (a single DICOM slice) | a one-slice volume at its position, so it overlays with volumes |
| `profiles.ml.boxes` (3-D) | Labels layer of box outlines: every slice in every plane shows the box, and so does 3-D; hovering shows label and score |
| `profiles.ml.boxes` (2-D) | Shapes rectangles with label and score |

Boxes are drawn by [napari-bboxes](https://github.com/Karol-G/napari-bboxes), which you can also use on its own for
boxes from any source.

Files larger than 1 GiB are read slice by slice and stay on disk. Labels are always loaded, so you can paint.

## Saving

`File > Save Selected Layer(s)` with a `.mla` name writes Image and Labels layers:

- A layer opened from a `.mla` file is written in its original array order with all its metadata: axes, values and
  label table, profiles, user metadata, source headers and history (plus an `edited` entry). Value scaling applied
  for display is undone when that is exact. The geometry comes from the layer's current transform, so a layer
  you moved is saved where it is shown.
- A new layer (for example labels you painted on an image) is written in the patient frame (LPS) when the viewer
  shows patient axes (`I, P, L` or `I, P, R`), so it lines up with the image in any other tool. Label names from
  the layer's features (`index`, `name`) become the label table.

## Development

```bash
pip install -e . --group dev
pytest
```

The tests check every voxel's world position for axial, sagittal, flipped and RAS-stored grids in both
conventions, oblique grids, labels, channels, time, single slices, boxes and save/load round trips.
`xvfb-run python scripts/verify_orientation.py OUTDIR` checks the picture on screen.

## License

Distributed under the terms of the MIT license.

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.
