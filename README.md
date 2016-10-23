# The SHOGUN machine learning toolbox
-------------------------------------

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

## Introduction
---------------
The Shogun Machine learning toolbox provides a wide range of *unified* and *efficient* Machine Learning (ML) methods. The toolbox seamlessly allows to easily combine multiple data representations, algorithm classes, and general purpose tools. This enables both rapid prototyping of data pipelines and extensibility in terms of new algorithms. We combine modern software architecture in C++ with both efficient low-level computing backends and cutting edge algorithm implementations to solve large-scale Machine Learning problems (yet) on single machines.

One of Shogun's most exciting features is that you can use the toolbox through a *unified* interface from C++, Python, Octave, R, Java, Lua, C#, etc. This not just means that we are independent of trends in computing languages, but it also lets you use Shogun as a vehicle to expose your algorithm to multiple communities. We use [SWIG](http://www.swig.org/) to enable *bidirectional* communication between C++ and target languages. Shogun runs under Linux/Unix, MacOS, Windows.

Originally focussing on large-scale kernel methods and bioinformatics (for a list of scientific papers mentioning Shogun, see [here](http://scholar.google.com/scholar?hl=en&q=shogun+toolbox&btnG=&as_sdt=1%2C33&as_sdtp=)), the toolbox saw massive extensions to other fields in recent years. It now offers features that span the whole space of Machine Learning methods, including many classical methods in classification, regression, dimensionality reduction, clustering, but also more advanced algorithm classes such as metric, multi-task, structured output, and online learning, as well as feature hashing, ensemble methods, and optimization, just to name a few. Shogun in addition contains a number of exclusive state-of-the art algorithms such as a wealth of efficient SVM implementations, Multiple Kernel Learning, kernel hypothesis testing, Krylov methods, etc. All algorithms are supported by a collection of general purpose methods for evaluation, parameter tuning, preprocessing, serialisation & I/O, etc; the resulting combinatorial possibilities are huge. See our [feature list](http://www.shogun-toolbox.org/page/features/) for more details.

The wealth of ML open-source software allows us to offer bindings to other sophisticated libraries including: [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)/[LibLinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), [SVMLight](http://svmlight.joachims.org/), [LibOCAS](http://cmp.felk.cvut.cz/~xfrancv/ocas/html/), [libqp](http://cmp.felk.cvut.cz/~xfrancv/libqp/html/), [VowpalWabbit](http://www.hunch.net/~vw/), [Tapkee](http://tapkee.lisitsyn.me/), [SLEP](http://www.public.asu.edu/~jye02/Software/SLEP/), [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/) and more. See our [list of integrated external libraries](http://www.shogun-toolbox.org/page/about/contributions).

Shogun got initiated in 1999 by [Soeren Sonnenburg](http://sonnenburgs.de/soeren) and [Gunnar Raetsch](http://www.raetschlab.org/) (that's where the name ShoGun originates from). It is now developed by a much larger Team cf. [website](http://shogun-toolbox.org/page/about/ourteam) and [AUTHORS](http://www.github.com/shogun-toolbox/shogun/wiki/AUTHORS), and would not have been possible without the patches and bug reports by various people. See [CONTRIBUTIONS](http://www.github.com/shogun-toolbox/shogun/wiki/CONTRIBUTIONS) for a detailed list. Statistics on Shogun's development activity can be found on [ohloh](https://www.openhub.net/p/shogun).

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
|r                 | *stable*                                                  |
|lua               | *beta* (many examples work, string typemaps are unstable, |
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

