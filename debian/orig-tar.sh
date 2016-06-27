#!/bin/sh

# called by uscan with '--upstream-version' <version> <file>

rm -rf shogun_$2 python-shogun_$2
rm -f python-shogun_$2.orig.tar*
tar xJf $3
pushd `pwd`
cd shogun_$2 || exit 1

rm -rf tests
rm -rf src/shogun
rm -rf benchmarks
rm -rf applications
rm -rf doc/tutorial
rm -rf `find examples/undocumented/ -maxdepth 1 -mindepth 1 ! -name '*python*'`
rm -rf `find src/interfaces/ -maxdepth 1 -mindepth 1 ! -name '*python*' | grep -v '/modular'`

popd

mv shogun_$2 python-shogun_$2
tar cf python-shogun_$2.orig.tar python-shogun_$2
xz -9 python-shogun_$2.orig.tar

rm -rf python-shogun_$2

# move to directory 'tarballs'
if [ -r .svn/deb-layout ]; then
  . .svn/deb-layout
  mv shogun_$2.orig.tar.xz $origDir
  echo "moved shogun_$2.orig.tar.xz to $origDir"
fi
