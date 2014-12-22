#!/bin/bash
#
# light-scrubber.sh
#
# Scrub stuff that depends or links against SVM^light, which we can't ship in
# tarballs used for several linux-distros.  Thanks to Dr. Sören Sonnenburg
# (sonney2k), the upstream author, for this.  The scriptlets have been
# slightly modified by me to preserve the original timestamps.
#
# TODO: Some next release or git-snapshot ships GPL'ed replacement of SVMlight
# called SVMbright.  The scriptlet isn't needed anymore when this will be.
#
# The following scriptlet is Copyright (C) 1999 - 2013  Dr. Sören Sonnenburg
# Modifications are Copyright (C) 2013 - 2014  Björn Esser
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
rm -rf	examples/*/*/{*light*,*_domainadaptationsvm_*}				\
	examples/undocumented/matlab_and_octave/tests/*light*			\
	src/shogun/classifier/svm/SVMLight.*					\
	src/shogun/classifier/svm/SVMLightOneClass.*				\
	src/shogun/regression/svr/SVRLight.*					\
	doc/md/LICENSE_SVMlight*

for _file in `grep -rl USE_SVMLIGHT .`
do
  sed -i.orig -e								\
	'/\#ifdef USE_SVMLIGHT/,/\#endif \/\/USE_SVMLIGHT/c \\' ${_file} &&	\
  touch -r ${_file}.orig ${_file} &&						\
  rm -rf ${_file}.orig
done

for _file in `find . -depth -name 'CMakeLists.txt'`
do
  sed -i.orig -e 's!.*_sv[mr]light_.*!!g' ${_file} &&				\
  touch -r ${_file}.orig ${_file} &&						\
  rm -rf ${_file}.orig
done

for _file in src/shogun/kernel/Kernel.{cpp,h}
do
  sed -i.orig -e '/^ \* EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS/,/ \* this program is free software/c\ * This program is free software; you can redistribute it and/or modify'	\
	 ${_file} &&								\
  touch -r ${_file}.orig ${_file} &&						\
  rm -rf ${_file}.orig
done

_file="src/interfaces/modular/Transfer_includes.i" &&				\
cp -a ${_file} ${_file}.orig &&							\
echo '%}' >> ${_file} &&							\
touch -r ${_file}.orig ${_file} &&						\
rm -rf ${_file}.orig

_file="src/interfaces/modular/Machine.i" &&					\
sed -i.orig -e 'd/.*CSVRLight.*/' ${_file} &&					\
touch -r ${_file}.orig ${_file} &&						\
rm -rf ${_file}.orig

_file="examples/undocumented/libshogun/" &&					\
_file="${_file}evaluation_cross_validation_locked_comparison.cpp" &&		\
sed -i.orig -e '/.*SVMLight.h>$/d' ${_file} &&					\
touch -r ${_file}.orig ${_file} &&						\
rm -rf ${_file}.orig
