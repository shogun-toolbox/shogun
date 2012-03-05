# **NOTE** First build a bare-bones shogun from Cygwin environment:
# ./configure --interfaces=cmdline_static --disable-bzip2 --disable-glpk --disable-gzip --disable-hdf5 --disable-json --disable-lzma --disable-lzo --disable-snappy --disable-svm-light --disable-xml
# make

project(shogun)
cmake_minimum_required(VERSION 2.8)

add_subdirectory(shogun)

