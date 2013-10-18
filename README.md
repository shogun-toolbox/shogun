# The SHOGUN machine learning toolbox
-------------------------------------

Develop branch build status:

[![Build Status](https://travis-ci.org/shogun-toolbox/shogun.png?branch=develop)](https://travis-ci.org/shogun-toolbox/shogun)

Buildbot: http://buildbot.shogun-toolbox.org/waterfall.

Quick links to this file:

* [Introduction](#introduction)
* [Interfaces](#interfaces)
* [Platforms](#platforms)
* [Directory Contents](#contents)
* [Applications](#applications)
* [License](#license)
* [Download](#download)
* [References](#references)

Other links that may be useful:

* See [INSTALL](INSTALL.md) for first steps on installation and running SHOGUN.
* See [README.developer](README_developer.md) for the developer documentation
* See [README.data](README_data.md) for how to download example data sets accompanying SHOGUN.

## Introduction
---------------

The machine learning toolbox's focus is on large scale kernel methods and
especially on Support Vector Machines (SVM) [1]. It provides a generic SVM
object interfacing to several different SVM implementations, among them the
state of the art LibSVM [2] and SVMlight [3].  Each of the SVMs can be
combined with a variety of kernels. The toolbox not only provides efficient
implementations of the most common kernels, like the Linear, Polynomial,
Gaussian and Sigmoid Kernel but also comes with a number of recent string
kernels as e.g. the Locality Improved [4], Fischer [5], TOP [6], Spectrum [7],
Weighted Degree Kernel (with shifts) [8, 9, 10]. For the latter the efficient
LINADD [10] optimizations are implemented.  Also SHOGUN offers the freedom of
working with custom pre-computed kernels.  One of its key features is the
*combined kernel* which can be constructed by a weighted linear combination
of a number of sub-kernels, each of which not necessarily working on the same
domain. An optimal sub-kernel weighting can be learned using Multiple Kernel
Learning [11, 12, 16]. Currently SVM 2-class classification and regression problems can be dealt
with. However SHOGUN also implements a number of linear methods like Linear
Discriminant Analysis (LDA), Linear Programming Machine (LPM), (Kernel)
Perceptrons and features algorithms to train hidden markov models.
The input feature-objects can be dense, sparse or strings, and
of types int/short/double/char. In addition, they can be converted into different feature types.
Chains of *preprocessors* (e.g. substracting the mean) can be attached to
each feature object allowing for on-the-fly pre-processing.

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
|r\_modular        | *pre-alpha* (SWIG does not properly handle reference counting and thus only for the brave, <br/> --disable-reference-counting to get it to work, but beware that it will leak memory; disabled by default)           |
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
the [INSTALL](INSTALL.md) file for generic and platform specific installation instructions).

## Directory Contents
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

## License
----------

Except for the files classifier/svm/Optimizer.{cpp,h},
classifier/svm/SVM_light.{cpp,h}, regression/svr/SVR_light.{cpp,h}
and the kernel caching functions in kernel/Kernel.{cpp,h}
which are (C) Torsten Joachims and follow a different
licensing scheme (cf. [LICENSE\_SVMlight](LICENSE_SVMlight.md)) SHOGUN is
generally licensed under the GPL version 3 or any later version (cf.
[LICENSE](LICENSE.md)) with code borrowed from various GPL compatible
libraries from various places (cf. [CONTRIBUTIONS](CONTRIBUTIONS.md)). See also
[LICENSE\_msufsort](LICENSE_msufsort.md) and  [LICENSE\tapkee](LICENSE_tapkee.md).

## Download
-----------

SHOGUN can be downloaded from http://www.shogun-toolbox.org and GitHub at
https://github.com/shogun-toolbox/shogun.

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
