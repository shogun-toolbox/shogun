#!/bin/bash
( cd ../../src
for e in ../examples/octave-modular/*.m
do
	echo -n "running $e .."
	if octave "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
	fi
done
)
