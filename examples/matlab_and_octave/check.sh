#!/bin/bash

INTERFACE=`grep '^INTERFACE' ../../src/.config | cut -f 2 -d '=' | tr -d ' '`

function errormsg
{
	echo
	echo $1
	echo
	echo "not running checks...."
	echo
	exit 1
}

if [ "${INTERFACE}" = "matlab" ]
then
	errormsg "matlab is not capable of setting proper exit codes try octave!"
fi

if [ "$INTERFACE" != "octave" ]
then
	errormsg "octave is not the configured interface!"
fi


for e in *.m
do
	echo -n "running $e .."
	if octave -p ../../src "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
	fi
done
