#!/bin/bash

( cd examples
for e in *.sg
do
	echo -n "running $e .."
	if ../../src/shogun $e >/dev/null 2>&1
	then
		echo " passed"
	else
		echo " failed"
	fi
done
)
