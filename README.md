# The SHOGUN machine learning toolbox
-------------------------------------

Unified and efficient Machine Learning.
See [ABOUT](https://github.com/shogun-toolbox/docs/blob/master/ABOUT.md) for a project description.

Develop branch build status:

[![Build Status](https://travis-ci.org/shogun-toolbox/shogun.png?branch=develop)](https://travis-ci.org/shogun-toolbox/shogun)
[![Coverage Status](https://coveralls.io/repos/shogun-toolbox/shogun/badge.png?branch=develop)](https://coveralls.io/r/shogun-toolbox/shogun?branch=develop)

Buildbot: http://buildbot.shogun-toolbox.org/waterfall.

Quick links to this file:

* [Quickstart](https://github.com/shogun-toolbox/shogun/wiki/QUICKSTART)
* [Introduction](#introduction)
* [Interfaces](#interfaces)
* [Platforms](#platforms)
* [Contents](#contents)
* [Applications](#applications)
* [License](#license)
* [Download](#download)
* [References](#references)

Other links that may be useful:

* See [INSTALL](https://github.com/shogun-toolbox/shogun/wiki/INSTALL) for first steps on installation and running SHOGUN.
* See [README.developer](https://github.com/shogun-toolbox/shogun/wiki/README_developer) for the developer documentation.
* See [README.data](https://github.com/shogun-toolbox/shogun/wiki/README_data) for how to download example data sets accompanying SHOGUN.
* See [README.cmake](https://github.com/shogun-toolbox/shogun/wiki/README_cmake) for setting particular build options with SHOGUN and cmake.

## Interfaces
-------------

SHOGUN is implemented in C++ and interfaces to Matlab(tm), R, Octave,
Java, C#, Ruby, Lua and Python.

The following table depicts the status of each interface available in SHOGUN:

|    Interface     |     Status                                                |
|:----------------:|-----------------------------------------------------------|
|python            | *mature* (no known problems)                              |
|octave            | *mature* (no known problems)                              |
|java              | *stable* (no known problems)                              |
|ruby              | *stable* (no known problems)                              |
|csharp            | *stable* (no known problems)                              |
|r                 | *beta*   (most examples work, static calls unavailable    |
|lua               | *alpha* (many examples work, string typemaps are unstable,|
                   |         overloaded methods unavailable)                   |
|perl              | *pre-alpha* (work in progress quality)                    |
|js                | *pre-alpha* (work in progress quality)                    |



Visit http://www.shogun-toolbox.org for further information.


## Platforms
------------

Debian GNU/Linux, Mac OSX and WIN32/CYGWIN are supported platforms (see
the [INSTALL](doc/md/INSTALL.md) file for generic and platform specific installation instructions).

## Contents
-----------

The following directories are found in the source distribution.

- *src* - source code.
- *data* - data sets (required for some examples / applications - these need to be downloaded
    separately via the download site or `git submodule update --init` from the root of the git checkout.
- *doc* - documentation (to be built using doxygen), ipython notebooks, and the PDF tutorial.
- *examples* - example files for all interfaces.
- *applications* - applications of SHOGUN.
- *benchmarks* - speed benchmarks.
- *tests* - unit and integration tests.
- *cmake* - cmake build scripts

## Applications
---------------

We have successfully used this toolbox to tackle the following sequence
analysis problems: Protein Super Family classification,
Splice Site Prediction, Interpreting the SVM Classifier,
Splice Form Prediction, Alternative Splicing and Promotor
Prediction. Some of them come with no less than 10
million training examples, others with 7 billion test examples.

## License
----------

Except for the files classifier/svm/Optimizer.{cpp,h},
classifier/svm/SVM_light.{cpp,h}, regression/svr/SVR_light.{cpp,h}
and the kernel caching functions in kernel/Kernel.{cpp,h}
which are (C) Torsten Joachims and follow a different
licensing scheme (cf. [LICENSE\_SVMlight](doc/md/LICENSE_SVMlight.md)) SHOGUN is
generally licensed under the GPL version 3 or any later version (cf.
[LICENSE](doc/md/LICENSE.md)) with code borrowed from various GPL compatible
libraries from various places (cf. [CONTRIBUTIONS](doc/md/CONTRIBUTIONS.md)). See also
[LICENSE\_msufsort](doc/md/LICENSE_msufsort.md) and  [LICENSE\_tapkee](doc/md/LICENSE_tapkee.md).

## Download
-----------

SHOGUN can be downloaded from http://www.shogun-toolbox.org and GitHub at
https://github.com/shogun-toolbox/shogun.

