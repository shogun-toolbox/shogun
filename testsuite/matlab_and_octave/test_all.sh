#!/bin/bash

DATAPATH='../data'

function test_all () {
	datapath="$1"
	echo "*** Testing in $datapath"
	sleep 1
	for file in $datapath; do
		echo -n "$file"
		echo -n -e "\t\t"

		interface=`grep INTERFACE ../../src/.config | awk '{print $3}'`
		if [ ${interface} != "octave" -a ${interface} != "matlab" ]; then
			echo "Unknown interface ${interface}"
			exit 1
		fi

		output=`./test_one.sh ${file} ${interface}`

		if [ $? -ne 0 ]; then
			echo 'ERROR'
			echo ${output}
		else
			echo 'OK'
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
