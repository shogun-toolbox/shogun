#!/bin/bash

status=0

for e in *.m
do
	echo -n "running $e .."
	if octave "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		status=1
	fi
done
exit $status
