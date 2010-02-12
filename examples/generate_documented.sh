#!/bin/bash

static_dirs=cmdline elwms matlab_and_octave python r
modular_dirs=octave_modular python_modular r_modular
lib_dirs=libshogun 

for d in static_dirs
do
	files=`ls -1 undocumented/$d`
	rm -rf documented/$d
	mkdir documented/$d

	for f in files
	do
		prefix=`echo $f | cut -f 1 -d '.'`
		if test -f descriptions/static/$prefix
		then
			cat descriptions/static/$prefix undocumented/$d/$f >documented/$d/$f
		else
			cat undocumented/$d/$f >documented/$d/$f
		fi
	done

	test -d documented/$d/graphical && \
		cp -r documented/$d/graphical undocumented/$d/graphical
done
