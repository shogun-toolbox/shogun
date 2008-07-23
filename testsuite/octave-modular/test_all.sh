#!/bin/bash

DATAPATH='../data'

function test_all () {
	datapath="$1"
	echo "*** Testing in $datapath"
	sleep 1
	for file in $datapath; do
		echo -n "$file"
		echo -n -e "\t\t"

		output=`./test_one.sh ${file}`
		ans=`echo $output | grep 'ans =' | awk '{print $NF}'`

		# thanks to matlab, 1 means ok and 0 means error
		if [ $? -ne 0 -o ${ans} -eq 0 ]; then
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
