#!/bin/bash

MATLAB="$1"

status=0

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
			status=1
		fi
	else
		if octave "$e" >/dev/null 2>&1
		then
			echo " OK"
		else
			echo " ERROR"
			status=1
		fi
	fi
done
exit $status
