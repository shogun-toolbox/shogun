#!/bin/bash

for e in *.sg
do
	echo -n "running $e .."
	if shogun $e >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
	fi
done
