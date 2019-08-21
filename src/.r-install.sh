#!/usr/bin/env bash

VERSION=`head -n 3 $1/NEWS | tail -n 1| awk '{ print $5 }'`
DATE=`head -n 1 $1/NEWS | cut -f 1 -d ' '`
RVERSION=`R --slave -e "with(R.version, cat(sprintf('%s.%s', major, minor)))"`
PLATFORM=`R --slave -e "cat(R.version\\$platform)"`
OSTYPE=`R --slave -e "cat(.Platform\\$OS.type)"`
DATE="`date '+%Y-%m-%d %H:%M:%S'`"

echo "Installing modular shogun interface for R"

rm -rf $2

mkdir -p $2/R
mkdir -p $2/inst/libs

for f in *.so
do
	cp $f $2/inst/libs/
done

for f in *.R
do
	cp $f $2/R/
done


cat >"$2/DESCRIPTION" <<EOF
Package: $2
Version: $VERSION
Date: $DATE
Title: The SHOGUN Machine Learning Toolbox
Author: Soeren Sonnenburg, Gunnar Raetsch
Maintainer: Soeren Sonnenburg <sonne@debian.org>
Depends: R (>= 2.10.0)
Suggests:
Description: SHOGUN - is a new machine learning toolbox with focus on large
        scale kernel methods and especially on Support Vector Machines (SVM) with focus
        to bioinformatics. It provides a generic SVM object interfacing to several
        different SVM implementations. Each of the SVMs can be combined with a variety
        of the many kernels implemented. It can deal with weighted linear combination
        of a number of sub-kernels, each of which not necessarily working on the same
        domain, where  an optimal sub-kernel weighting can be learned using Multiple
        Kernel Learning.  Apart from SVM 2-class classification and regression
        problems, a number of linear methods like Linear Discriminant Analysis (LDA),
        Linear Programming Machine (LPM), (Kernel) Perceptrons and also algorithms to
        train hidden markov models are implemented. The input feature-objects can be
        dense, sparse or strings and of type int/short/double/char and can be converted
        into different feature types. Chains of preprocessors (e.g.  substracting the
        mean) can be attached to each feature object allowing for on-the-fly
        pre-processing.
License: BSD 3-clause license (see LICENSE file).
URL: http://www.shogun-toolbox.org
LazyData: true
EOF

cat >"$2/NAMESPACE" <<EOF
useDynLib(shogun, .registration = TRUE)
EOF

cat >"$2/R/init.R" <<EOF
.packageName <- "$2"
#$2 <- function(...) .External("$2",...,PACKAGE="$2")

# Load the shogun dynamic library at startup.
#
.First.lib <- function(lib,pkg)
{
	cat(paste("\nWelcome! This is SHOGUN version $VERSION\n"))
EOF

for f in *.so
do
	echo "library.dynam(\"`basename $f .so`\", pkg, lib)" >> "$2/R/init.R"
done

cat >>"$2/R/init.R" <<EOF
}
# Unload the library.
#
.Last.lib <- function(lib)
{
EOF

for f in *.so
do
	echo "library.dynam.unload(\"$f\", lib)" >> "$2/R/init.R"
done

cat >>"$2/R/init.R" <<EOF
}

# Because in packages with namespaces .First.lib will not be loaded
# one needs another functions called .onLoad resp. .onUnload
#
.onLoad <- function(lib, pkg) .First.lib(lib,pkg)
.onUnload <- function(lib) .Last.lib(lib)
EOF

R CMD INSTALL --no-multiarch --with-keep.source shogun