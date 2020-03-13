# The SHOGUN machine learning toolbox
-------------------------------------

Unified and efficient Machine Learning since 1999.

Latest release:

[![Release](https://img.shields.io/github/release/shogun-toolbox/shogun.svg)](https://github.com/shogun-toolbox/shogun/releases/latest)

Cite Shogun:

[![DOI](https://zenodo.org/badge/1555094.svg)](https://zenodo.org/badge/latestdoi/1555094)

Develop branch build status:

[![Build status](https://dev.azure.com/shogunml/shogun/_apis/build/status/shogun-CI)](https://dev.azure.com/shogunml/shogun/_build/latest?definitionId=-1)
[![codecov](https://codecov.io/gh/shogun-toolbox/shogun/branch/develop/graph/badge.svg)](https://codecov.io/gh/shogun-toolbox/shogun)

Donate to Shogun via NumFocus:

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](http://numfocus.org)


Buildbot: https://buildbot.shogun.ml.

 * See [doc/readme/ABOUT.md](doc/readme/ABOUT.md) for a project description.
 * See [doc/readme/INSTALL.md](doc/readme/INSTALL.md) for installation instructions.
 * See [doc/readme/INTERFACES.md](doc/readme/INTERFACES.md) for calling Shogun from its interfaces.
 * See [doc/readme/EXAMPLES.md](doc/readme/EXAMPLES.md) for details on creating API examples.
 * See [doc/readme/DEVELOPING.md](doc/readme/DEVELOPING.md) for how to hack Shogun.
 
 * See [API examples](http://shogun.ml/examples) for all interfaces.
 * See [the wiki](https://github.com/shogun-toolbox/shogun/wiki/) for extended developer information.

## Interfaces
-------------

Shogun is implemented in C++ and offers automatically generated, unified interfaces to Python, Octave, Java / Scala, Ruby, C#, R, Lua. We are currently working on adding more languages including JavaScript, D, and Matlab.

|    Interface     |     Status                                                |
|:----------------:|-----------------------------------------------------------|
|Python            | *mature* (no known problems)                              |
|Octave            | *mature* (no known problems)                              |
|Java/Scala        | *stable* (no known problems)                              |
|Ruby              | *stable* (no known problems)                              |
|C#                | *stable* (no known problems)                              |
|R                 | Currently disabled due to a [swig bug](https://github.com/swig/swig/issues/1660) |
|Perl              | *pre-alpha* (work in progress quality)                    |
|JS                | *pre-alpha* (work in progress quality)                    |

See [our website](http://shogun.ml/examples) for examples in all languages.

## Platforms
------------

Shogun is supported under GNU/Linux, MacOSX, FreeBSD, and Windows.

## Directory Contents
---------------------

The following directories are found in the source distribution.
Note that some folders are submodules that can be checked out with
`git submodule update --init`.

- *src* - source code, separated into C++ source and interfaces
- *doc* - readmes (doc/readme, submodule), Jupyter notebooks, cookbook (API examples), licenses
- *examples* - example files for all interfaces
- *data* - data sets (submodule, required for examples)
- *tests* - unit tests and continuous integration of interface examples
- *applications* - applications of SHOGUN (outdated)
- *benchmarks* - speed benchmarks
- *cmake* - cmake build scripts

## License
----------
Shogun is distributed under [BSD 3-clause license](doc/license/LICENSE.md), with
optional GPL3 components.
See [doc/licenses](doc/license) for details.
