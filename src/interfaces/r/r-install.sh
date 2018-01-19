#!/usr/bin/env bash

VERSION=`head -n 3 ../../../../NEWS | tail -n 1| awk '{ print $5 }'`
DATE=`head -n 1 ../../../../NEWS | cut -f 1 -d ' '`
RVERSION=`R --slave -e "with(R.version, cat(sprintf('%s.%s', major, minor)))"`
PLATFORM=`R --slave -e "cat(R.version\\$platform)"`
OSTYPE=`R --slave -e "cat(.Platform\\$OS.type)"`
DATE="`date '+%Y-%m-%d %H:%M:%S'`"
PKGFILE="$1/$2/Meta/package.rds"
SAVERDS="$4"

echo "x=structure(list(DESCRIPTION = c(Package='$2',\
		Version=\"$VERSION\",\
		Date=\"$DATE\",\
		Title=\"The SHOGUN Machine Learning Toolbox\",\
		Author=\"Shogun Team\",\
		Maintainer=\"shogun-team@shogun-toolbox.org\",\
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

# R-MODULAR
echo "Installing modular shogun interface for R"

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
