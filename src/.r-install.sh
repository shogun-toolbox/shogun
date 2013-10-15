#!/usr/bin/env bash

VERSION=`head -n 3 ../../../../NEWS | tail -n 1| awk '{ print $5 }'`
DATE=`head -n 1 ../../../../NEWS | cut -f 1 -d ' '`
RVERSION=`R --slave -e "with(R.version, cat(sprintf('%s.%s', major, minor)))"`
PLATFORM=`R --slave -e "cat(R.version\\$platform)"`
OSTYPE=`R --slave -e "cat(.Platform\\$OS.type)"`
DATE="`date '+%Y-%m-%d %H:%M:%S'`"
PKGFILE="$1/$2/Meta/package.rds"
SAVERDS="$4"

cat >"$1/$2/DESCRIPTION" <<EOF
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
License: GPL Version 3 or later.
URL: http://www.shogun-toolbox.org
Built: $RVERSION; $PLATFORM; $OSTYPE;
EOF

echo "x=structure(list(DESCRIPTION = c(Package='$2',\
		Version=\"$VERSION\",\
		Date=\"$DATE\",\
		Title=\"The SHOGUN Machine Learning Toolbox\",\
		Author=\"Soeren Sonnenburg, Gunnar Raetsch\",\
		Maintainer=\"sonne@debian.org\",\
		Depends=\"R (>= $RVERSION)\", \
		Built=\"R $RVERSION; ; $DATE\"),\
		Built = list(R=\"$RVERSION\", Platform=\"$PLATFORM\", Date=\"$DATE\", OStype=\"$OSTYPE\"),\
		Rdepends = list(name='R', op='>=', version='2.10'),\
		Rdepends2 = list(list(name='R', op='>=', version='2.10')),\
		Depends = list(),\
		Suggests = list(),\
		Imports = list()),\
		class = 'packageDescription2');\
		$SAVERDS(x, \"$PKGFILE\")" | R --no-save

# R STATIC
if test "$2" = "sg" || test "$2" = "elwms"
then
echo "Installing static sg/elwms interface for R"
cat >"$1/$2/NAMESPACE" <<EOF
export(sg)
EOF

cat >"$1/$2/R/$2" <<EOF
.packageName <- "$2"
# The purpose of this file is to supply no functionality
# except easier access functions in R for external C
# function calls.
#
# For example instead of typing
#
#     > .External("$2", "send_command", "blah")
#
# one can simply type
#
#     > send_command(blah)
#
# where > is the R prompt.

# interface $2(arg1,arg2,...) as w/ matlab/octave/python
#
$2 <- function(...) .External("$2",...,PACKAGE="$2")


# R specific interface

# Generic functions
#
send_command <- function(x) .External("$2","send_command",x,PACKAGE="$2")
set_features <- function(x,y) .External("$2","set_features",x,y,PACKAGE="$2")
add_features <- function(x,y) .External("$2","add_features",x,y,PACKAGE="$2")
set_labels <- function(x,y) .External("$2","set_labels", x,y,PACKAGE="$2")
get_kernel_matrix <- function() .External("$2","get_kernel_matrix",PACKAGE="$2")

# SVM functions
#
svm_classify <- function() .External("$2","svm_classify",PACKAGE="$2")
get_svm <- function() .External("$2","get_svm",PACKAGE="$2")
get_subkernel_weights <- function() .External("$2","get_subkernel_weights",PACKAGE="$2")

# HMM functions
#
get_hmm <- function() .External("$2","get_hmm",PACKAGE="$2")

# Load the shogun dynamic library at startup.
#
.First.lib <- function(lib,pkg)
{
	cat(paste("\nWelcome! This is SHOGUN version $VERSION\n"))
	library.dynam("$2",pkg,lib)
}

# Unload the library.
#
.Last.lib <- function(lib) library.dynam.unload("$2", libpath=lib)

# Because in packages with namespaces .First.lib will not be loaded
# one needs another functions called .onLoad resp. .onUnload
#

.onLoad <- function(lib, pkg) .First.lib(lib,pkg)
.onUnload <- function(lib) .Last.lib(lib)
EOF

# R-MODULAR
else
echo "Installing modular shogun interface for R"

cat >"$1/$2/NAMESPACE" <<EOF
useDynLib(modshogun, .registration = TRUE)
EOF

cat >"$1/$2/R/$2" <<EOF
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
	echo "library.dynam(\"`basename $f .so`\", pkg, lib)" >> "$1/$2/R/$2"
done

for f in *.RData
do
	echo "load(paste(lib, \"/\", \"$2\", \"/R/\", \"`basename $f`\", sep=''), envir=.GlobalEnv)" >> "$1/$2/R/$2"
	echo "cacheMetaData(1)" >> "$1/$2/R/$2"
done
cat >>"$1/$2/R/$2" <<EOF
}
# Unload the library.
#
.Last.lib <- function(lib)
{
EOF

for f in *.$3
do
	echo "library.dynam.unload(\"$f\", lib)" >> "$1/$2/R/$2"
done

cat >>"$1/$2/R/$2" <<EOF
}

# Because in packages with namespaces .First.lib will not be loaded
# one needs another functions called .onLoad resp. .onUnload
#
.onLoad <- function(lib, pkg) .First.lib(lib,pkg)
.onUnload <- function(lib) .Last.lib(lib)
EOF
fi
