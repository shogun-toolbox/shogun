#!/usr/bin/env bash

PACKAGE_NAME=$1
LIB_EXTENSION=$2

echo "Installing modular shogun interface for R"

# clean up
rm -rf ${PACKAGE_NAME}

# create package directories
mkdir -p ${PACKAGE_NAME}/R
mkdir -p ${PACKAGE_NAME}/inst/libs

cp *${LIB_EXTENSION} ${PACKAGE_NAME}/inst/libs/
cp *.R ${PACKAGE_NAME}/R/
cp DESCRIPTION ${PACKAGE_NAME}/
cp NAMESPACE.i ${PACKAGE_NAME}/

R CMD INSTALL --no-multiarch --with-keep.source --byte-compile ${PACKAGE_NAME}
