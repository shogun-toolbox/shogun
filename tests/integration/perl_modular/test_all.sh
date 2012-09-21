#!/usr/bin/env bash

if test -z "$PYTHON" 
if test -z "$PERL" 
then
	PYTHON=python
	PERL=perl
fi

DATAPATH='../data'

function test_all () {
	datapath="$1"

	if echo "$datapath" | grep -q '\.\./data/tests'
	then
		continue
	fi

	echo "*** Testing in $datapath"
	sleep 1
	for file in $datapath; do
		echo -n "$file"
		echo -n -e "\t\t"

		output=`${PYTHON} test_one.py "$file"`
		output=`${PERL} test_one.py "$file"`
		ret=$?

		if [ $ret -eq 0 ] ; then
			echo 'OK'
		else
			echo 'ERROR'
			echo $output
		fi
	done
	sleep 1
	echo
}

if [ -n "$1" ]; then
	test_all "$DATAPATH/$1/*.m"
else
	for i in $DATAPATH/*; do
		test_all "$i/*.m"
	done
fi
