#!/bin/bash

rm -f error.log

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
		echo "================================================================================" >>error.log
		echo " error in $e ">>error.log
		echo "================================================================================" >>error.log
		${PYTHON} "$e" >>error.log 2>&1
		echo "================================================================================" >>error.log
		echo >>error.log
		echo >>error.log
	fi
done

test -f error.log && ( cat error.log ; exit 1 )
exit 0
