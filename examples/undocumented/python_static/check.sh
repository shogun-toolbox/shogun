#!/bin/bash
if [ -z "${PYTHON}" ]
then
	PYTHON=python
fi

(
for e in *.py
do
	echo -n "running $e .."
	if ${PYTHON} "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
	fi
done
)
