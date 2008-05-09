#!/bin/bash

OCTAVE=`grep '^INTERFACE' ../src/.config | cut -f 2 -d '=' | tr -d ' '`

if [ "$OCTAVE" = matlab ]
then
	echo
	echo "matlab is not capable of setting proper exit codes try octave!"
	echo
	echo "not running checks...."
	echo
	exit 1
fi

( cd ../src
for e in ../matlab_and_octave/examples/*.m
do
	echo -n "running $e .."
	if cat "$e" | ${OCTAVE} >/dev/null 2>&1
	then
		echo " passed"
	else
		echo " failed"
	fi
done
)
