#!/bin/bash

status=0

if [ -z "${PYTHON}" ]
then
	PYTHON=python
fi

for e in *.py
do
	echo -n "running $e .."
	if ${PYTHON} "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		status=1
	fi
done
exit $status
