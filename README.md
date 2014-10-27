# The SHOGUN machine learning toolbox
-------------------------------------

Develop branch build status:

[![Build Status](https://travis-ci.org/shogun-toolbox/shogun.png?branch=develop)](https://travis-ci.org/shogun-toolbox/shogun)
[![Coverage Status](https://coveralls.io/repos/shogun-toolbox/shogun/badge.png?branch=develop)](https://coveralls.io/r/shogun-toolbox/shogun?branch=develop)

Buildbot: http://buildbot.shogun-toolbox.org/waterfall.

Quick links to this file:

* [Introduction](#introduction)
* [Interfaces](#interfaces)
* [Platforms](#platforms)
* [Contents](#contents)
* [Applications](#applications)
* [Download](#download)
* [License](#license)
* [Contributions](#contributions)
* [References](#references)

Other links that may be useful:

* See [QUICKSTART](https://github.com/shogun-toolbox/shogun/wiki/QUICKSTART) for first steps on installation and running SHOGUN.
* See [README\_developer](https://github.com/shogun-toolbox/shogun/wiki/README_developer) for the developer documentation.
* See [README\_data](https://github.com/shogun-toolbox/shogun/wiki/README_data) for how to download example data sets accompanying SHOGUN.
* See [README\_cmake](https://github.com/shogun-toolbox/shogun/wiki/README_cmake) for setting particular build options with SHOGUN and cmake.

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
|python\_modular   | *mature* (no known problems)                              |
|octave\_modular   | *mature* (no known problems)                              |
|java\_modular     | *stable* (no known problems; not all examples are ported) |
|ruby\_modular     | *stable* (no known problems; only few examples ported)    |
|csharp\_modular   | *stable* (no known problems; not all examples are ported) |
|lua\_modular      | *alpha* (some examples work, string typemaps are unstable |
|perl\_modular     | *pre-alpha* (work in progress quality)                    |
|r\_modular        | *stable* (no known problems; not all examples are ported) |
|octave\_static    | *mature* (no known problems)                                |
|matlab\_static    | *mature* (no known problems)                                |
|python\_static    | *mature* (no known problems)                                |
|r\_static         | *mature* (no known problems)                                |
|libshogun\_static | *mature* (no known problems)                                |
|cmdline\_static   | *stable* (but some data types incomplete)                 |
|elwms\_static     | this is the eierlegendewollmilchsau interface, a chimera that in one file interfaces with python, octave, r, matlab and provides the run\_python command to run code in python using the in octave,r,matlab available variables, etc)    |

Visit http://www.shogun-toolbox.org/doc/en/current for further information.


## Platforms
------------

Debian GNU/Linux, Mac OSX and WIN32/CYGWIN are supported platforms (see
the [QUICKSTART](https://github.com/shogun-toolbox/shogun/wiki/QUICKSTART) file for generic installation instructions).

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
analysis problems: Protein Super Family classification[6],
Splice Site Prediction [8, 13, 14], Interpreting the SVM Classifier [11, 12],
Splice Form Prediction [8], Alternative Splicing [9] and Promotor
Prediction [15]. Some of them come with no less than 10
million training examples, others with 7 billion test examples.

## Download
-----------

SHOGUN can be downloaded from http://www.shogun-toolbox.org and GitHub at
https://github.com/shogun-toolbox/shogun.

## License
----------

Except for the files classifier/svm/Optimizer.{cpp,h},
classifier/svm/SVM_light.{cpp,h}, regression/svr/SVR_light.{cpp,h}
and the kernel caching functions in kernel/Kernel.{cpp,h}
which are (C) Torsten Joachims and follow a different
licensing scheme (cf. [LICENSE\_SVMlight](https://github.com/shogun-toolbox/shogun/wiki/LICENSE_SVMlight)) SHOGUN is
generally licensed under the GPL version 3 or any later version (cf.
[LICENSE](https://github.com/shogun-toolbox/shogun/wiki/LICENSE)) with code borrowed from various GPL compatible
libraries from various places (cf. [CONTRIBUTIONS](https://github.com/shogun-toolbox/shogun/wiki/CONTRIBUTIONS)). See also
[LICENSE\_msufsort](https://github.com/shogun-toolbox/shogun/wiki/LICENSE_msufsort) and  [LICENSE\_tapkee](https://github.com/shogun-toolbox/shogun/wiki/LICENSE_tapkee).

## Contributions
----------------

We include numerous lines of code from various standalone free libraries including the following:

- [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) by Chih-Chung Chang and Chih-Jen Lin
- [LibLinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) by Xiang-Rui Wang and Chih-Jen Lin
- [SLEP](http://www.public.asu.edu/~jye02/Software/SLEP/) by J. Liu, S. Ji and J. Ye
- [MALSAR](http://www.public.asu.edu/~jye02/Software/MALSAR/) by J. Zhou, J. Chen and J. Y
- LaRank by A. Bordes
- [GPBT](http://dm.unife.it/gpdt/) by T. Serafini, L. Zanni and G. Zanghirati
- [LibOCAS](http://cmp.felk.cvut.cz/~xfrancv/ocas/html/) by V. Franc and S. Sonnenburg
- [SVMLin](http://vikas.sindhwani.org/svmlin.html) by V. Sindhwani
- [SGDSVM](http://leon.bottou.org/projects/sgd) by Leon Bottou
- [Vowpal](http://hunch.net/~vw/) Wabbit by John Langford
- [Cover](http://hunch.net/~jl/projects/cover_tree/cover_tree.html) Tree for Nearest Neighbour calculation by John Langford
- [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/) by Carl Edward Rasmussen and Hannes Nickisch

as well as take inspiration from many other libraries:

- [LMNN](http://www.cse.wustl.edu/~kilian/code/lmnn/lmnn.html) by Kilian Q. Weinberger
- [Hidden Markov Support Vector Machines](http://mloss.org/software/view/250/) by Georg Zeller, Gunnar Raetsch and Pramod Mudrakarta
- [Matlab Toolbox for Dimensionality Reduction](http://homepage.tudelft.nl/19j49/Matlab_Toolbox_for_Dimensionality_Reduction.html) by Laurens van der Maaten

Please let us know if we missed your name in this page, we will do our best to acknowledge your contributions.

## References
-------------

[1] C. Cortes and V.N. Vapnik.  Support-vector networks.
	Machine Learning, 20(3):273--297, 1995.

[2] J. Liu, S. Ji, and J. Ye. SLEP: Sparse Learning with Efficient Projections. Arizona State University, 2009.
	http://www.public.asu.edu/~jye02/Software/SLEP.

[3] C.C. Chang and C.J. Lin.  Libsvm: Introduction and benchmarks.
	Technical report, Department of Computer Science and Information
	Engineering, National Taiwan University, Taipei, 2000.

[4] T. Joachims. Making large-scale SVM learning practical. In B.~Schoelkopf,
	C.J.C. Burges, and A.J. Smola, editors, Advances in Kernel Methods -
	Support Vector Learning, pages 169--184, Cambridge, MA, 1999. MIT Press.

[5] A.Zien, G.Raetsch, S.Mika, B.Schoelkopf, T.Lengauer, and K.-R.
	Mueller. Engineering Support Vector Machine Kernels That Recognize
	Translation Initiation Sites. Bioinformatics, 16(9):799-807, September 2000.

[6] T.S. Jaakkola and D.Haussler.Exploiting generative models in
	discriminative classifiers. In M.S. Kearns, S.A. Solla, and D.A. Cohn,
	editors, Advances in Neural Information Processing Systems, volume 11,
	pages 487-493, 1999.

[7] K.Tsuda, M.Kawanabe, G.Raetsch, S.Sonnenburg, and K.R. Mueller.
	A new discriminative kernel from probabilistic models.
	Neural Computation, 14:2397--2414, 2002.

[8] C.Leslie, E.Eskin, and W.S. Noble. The spectrum kernel: A string kernel
	for SVM protein classification. In R.B. Altman, A.K. Dunker, L.Hunter,
	K.Lauderdale, and T.E. Klein, editors, Proceedings of the Pacific
	Symposium on Biocomputing, pages 564-575, Kaua'i, Hawaii, 2002.

[9] G.Raetsch and S.Sonnenburg. Accurate Splice Site Prediction for
	Caenorhabditis Elegans, pages 277-298. MIT Press series on Computational
	Molecular Biology. MIT Press, 2004.

[10] G.Raetsch, S.Sonnenburg, and B.Schoelkopf. RASE: recognition of
	alternatively spliced exons in c. elegans. Bioinformatics,
	21:i369--i377, June 2005.

[11] S.Sonnenburg, G.Raetsch, and B.Schoelkopf. Large scale genomic sequence
	SVM classifiers. In Proceedings of the 22nd International Machine Learning
	Conference. ACM Press, 2005.

[12] S.Sonnenburg, G.Raetsch, and C.Schaefer. Learning interpretable SVMs
	for biological sequence classification. In RECOMB 2005, LNBI 3500,
	pages 389-407. Springer-Verlag Berlin Heidelberg, 2005.

[13] G.Raetsch, S.Sonnenburg, and C.Schaefer. Learning Interpretable SVMs
	for Biological Sequence Classification. BMC Bioinformatics, Special Issue
	from NIPS workshop on New Problems and Methods in Computational Biology
	Whistler, Canada, 18 December 2004, 7:(Suppl. 1):S9, March 2006.

[14] S. Sonnenburg. New methods for splice site recognition. Master's thesis,
	Humboldt University, 2002. Supervised by K.R. Mueller H.D. Burkhard and
	G. Raetsch.

[15] S. Sonnenburg, G. Raetsch, A. Jagota, and K.R. Mueller. New methods for
	splice-site recognition. In Proceedings of the International Conference on
	Artifical Neural Networks, 2002. Copyright by Springer.

[16] S. Sonnenburg, A. Zien, and G. Raetsch. ARTS: Accurate Recognition of
	Transcription Starts in Human. 2006.

[17] S. Sonnenburg, G. Raetsch, C.Schaefer, and B.Schoelkopf, Large Scale
	Multiple Kernel Learning, Journal of Machine Learning Research, 2006,
	K.Bennett and E.P. Hernandez Editors.

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/3e5ff04ff56513867eedb5b2f4261702 "githalytics.com")](http://githalytics.com/shogun-toolbox/shogun)
