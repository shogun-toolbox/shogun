#!/bin/bash
( cd ../src
for e in ../octave-modular/examples/*.m
do
	echo -n "running $e .."
	if octave "$e" >/dev/null 2>&1
	then
		echo " passed"
	else
		echo " failed"
	fi
done
)
