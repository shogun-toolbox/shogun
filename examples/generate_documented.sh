#!/bin/bash

static_dirs="cmdline elwms matlab_and_octave python r"
modular_dirs="octave_modular python_modular r_modular"
lib_dirs=libshogun 

for d in ${static_dirs}
do
	files=`ls -1 undocumented/$d | grep -v check.sh`
	rm -rf documented/$d
	mkdir documented/$d

	for f in $files
	do
		doc=`echo $f | cut -f 1 -d '.'`.txt
		echo undocumented/$d/$f
		if test -f undocumented/$d/$f
		then
			if test -f descriptions/static/$doc
			then
				cat descriptions/static/$doc undocumented/$d/$f >documented/$d/$f
			else
				cat undocumented/$d/$f >documented/$d/$f
			fi
		fi
	done

	test -d undocumented/$d/graphical && \
		( mkdir documented/$d/graphical &&  \
		cp undocumented/$d/graphical/* documented/$d/graphical/ )
done

for d in ${modular_dirs}
do
	files=`ls -1 undocumented/$d | grep -v check.sh`
	rm -rf documented/$d
	mkdir documented/$d

	for f in $files
	do
		doc=`echo $f | cut -f 1 -d '.'`.txt
		if test -f undocumented/$d/$f
		then
			if test -f descriptions/modular/$doc
			then
				cat descriptions/modular/$doc undocumented/$d/$f >documented/$d/$f
			else
				cat undocumented/$d/$f >documented/$d/$f
			fi
		fi
	done

	test -d undocumented/$d/graphical && \
		( mkdir documented/$d/graphical &&  \
		cp undocumented/$d/graphical/* documented/$d/graphical/ )
done

exit 0
