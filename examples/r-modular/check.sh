#!/bin/bash
for e in *.R
do
	echo -n "running $e .."
	if R --no-restore --no-save --no-readline --slave -f "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
	fi
done
