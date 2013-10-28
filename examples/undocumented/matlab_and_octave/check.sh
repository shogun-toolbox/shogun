#!/usr/bin/env bash

MATLAB="$1"

rm -f error.log

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
			cat "$e" | matlab -nojvm -nodesktop -nodisplay >>error.log
		fi
	else
		if octave "$e" >/dev/null 2>&1
		then
			echo " OK"
		else
			echo " ERROR"
			echo "================================================================================" >>error.log
			echo " error in $e ">>error.log
			echo "================================================================================" >>error.log
			octave "$e" >>error.log 2>&1
			echo "================================================================================" >>error.log
			echo >>error.log
			echo >>error.log
		fi
	fi
done

if test -f error.log
then
	cat error.log
	exit 1
fi

exit 0
