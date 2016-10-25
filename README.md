# The SHOGUN machine learning toolbox
-------------------------------------

Unified and efficient Machine Learning since 1999.

Develop branch build status:

[![Build Status](https://travis-ci.org/shogun-toolbox/shogun.png?branch=develop)](https://travis-ci.org/shogun-toolbox/shogun)
[![Coverage Status](https://coveralls.io/repos/shogun-toolbox/shogun/badge.png?branch=develop)](https://coveralls.io/r/shogun-toolbox/shogun?branch=develop)

Buildbot: http://buildbot.shogun-toolbox.org/waterfall.

 * See [doc/readme/ABOUT.md](https://github.com/shogun-toolbox/docs/blob/master/ABOUT.md) for a project description.
 * See [doc/readme/INSTALL.md](https://github.com/shogun-toolbox/docs/blob/master/ABOUT.md) for installation instructions.
 * See [doc/readme/INTERFACES.md](https://github.com/shogun-toolbox/docs/blob/master/INTERFACE.md) for calling Shogun from its interfaces.
 * See [the cookbook](http://shogun.ml/cookbook/latest/) for API examples for all interfaces.
 * See [the wiki](https://github.com/shogun-toolbox/shogun/wiki/) for developer information.
   * [doc/wiki/README.developer](https://github.com/shogun-toolbox/shogun/wiki/README_developer)
   * [doc/wiki/README.data](https://github.com/shogun-toolbox/shogun/wiki/README_data)
   * [doc/wiki/README.cmake](https://github.com/shogun-toolbox/shogun/wiki/README_cmake)

Quick links for this file:
* [Introduction](#introduction)
* [Interfaces](#interfaces)
* [Platforms](#platforms)
* [License](#license)


## Interfaces
-------------

Shogun is implemented in C++ and interfaces to Python, octave, java, ruby, C#, R, Lua, Perl. JavaScript and Matlab are planned to be (re-)introduced soon.

|    Interface     |     Status                                                |
|:----------------:|-----------------------------------------------------------|
|python            | *mature* (no known problems)                              |
|octave            | *mature* (no known problems)                              |
|java              | *stable* (no known problems)                              |
|ruby              | *stable* (no known problems)                              |
|csharp            | *stable* (no known problems)                              |
|r                 | *beta*   (most examples work, static calls unavailable    |
|lua               | *alpha* (many examples work, string typemaps are unstable,
                             overloaded methods unavailable)                   |
|perl              | *pre-alpha* (work in progress quality)                    |
|js                | *pre-alpha* (work in progress quality)                    |

See our website for examples in all languages.

## Platforms
------------

Shogun is supported under GNU/Linux, MacOSX, FreeBSD, and Windows.
See our buildfarm

## Directory Contents
---------------------

The following directories are found in the source distribution.
Note that some folders are submodules that can be checked out with
`git submodule update --init`.

- *src* - source code.
- *doc* - readmes (doc/reamde, submodule), ipython notebooks, cookbook (api examples), licenses
- *examples* - example files for all interfaces.
- *data* - data sets (submodule required for some examples / applications)
- *tests* - unit and integration tests.
- *applications* - applications of SHOGUN.
- *benchmarks* - speed benchmarks.
- *cmake* - cmake build scripts

## License
----------
Shogun is generally licensed under the GPL3, with
code borrowed from various external libraries, and optional
parts that are neither compatible with GPL nor BSD.
It is possible to compile a BSD3 compatible build of Shogun.

See doc/licenses for details.

