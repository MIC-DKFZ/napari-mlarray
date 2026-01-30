# napari-mlarray

[![License MIT](https://img.shields.io/pypi/l/napari-mlarray.svg?color=green)](https://github.com/MIC-DKFZ/napari-mlarray/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-mlarray.svg?color=green)](https://pypi.org/project/napari-mlarray)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-mlarray.svg?color=green)](https://python.org)
[![tests](https://github.com/MIC-DKFZ/napari-mlarray/workflows/tests/badge.svg)](https://github.com/MIC-DKFZ/napari-mlarray/actions)
[![codecov](https://codecov.io/gh/MIC-DKFZ/napari-mlarray/branch/main/graph/badge.svg)](https://codecov.io/gh/MIC-DKFZ/napari-mlarray)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-mlarray)](https://napari-hub.org/plugins/napari-mlarray)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A reader/writer Napari plugin for [MLArray](https://github.com/MIC-DKFZ/mlarray) images. MLArray is an array format specialized for Machine Learning with Blosc2 backend and standardized metadata. 

## Installation

You can install `napari-mlarray` via [pip]:

```
pip install napari-mlarray
```

If napari is not already installed, you can install `napari-mlarray` with napari and Qt via:

```
pip install "napari-mlarray[all]"
```



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-mlarray" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.