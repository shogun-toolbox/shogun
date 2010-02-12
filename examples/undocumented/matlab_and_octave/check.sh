#!/bin/bash

MATLAB="$1"

for e in *.m
do
	echo -n "running $e .."

	if [ -n "$MATLAB" ]
	then
		if cat "$e" | matlab -nojvm -nodesktop -nodisplay >/dev/null
		then
			echo " OK"
		else
			echo " ERROR"
		fi
	else
		if octave "$e" >/dev/null 2>&1
		then
			echo " OK"
		else
			echo " ERROR"
		fi
	fi
done
