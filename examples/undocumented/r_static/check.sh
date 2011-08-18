#!/bin/bash

status=0

for e in *.R
do
	echo -n "running $e .."
	if R --slave -f "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		status=1
	fi
done

exit $status
