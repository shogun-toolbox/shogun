#!/usr/bin/env bash

DATAPATH='../data'
exitcode=0

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

		if grep -q $file ../blacklist
		then
			echo 'SKIPPING'
		else
			output=`./test_one.sh ${file} ${interface}`

			if [ "${interface}" == "octave" ]; then
				ans=`echo $output | grep 'ans =' | awk '{print $NF}'`
			else # matlab has '>>' as last element in output
				ans=`echo $output | grep 'ans =' | awk '{print $(NF-1)}'`
			fi

			if [ -z ${ans} ]; then
				ans=0
			fi

			# thanks to matlab, 1 means ok and 0 means error
			if [ "$?" -ne 0 -o "${ans}" -eq 0 ]; then
				exitcode=1
				echo ERROR
				echo ${output}
			else
				echo OK
			fi
		fi
	done
	sleep 1
	echo
}

interface=${2-octave}

if [ "${interface}" != "octave" -a "${interface}" != "matlab" ]; then
	echo "Unknown interface ${interface}"
	exit 1
fi

if [ -n "$1" -a "$1" != "-" ]; then
	test_all "$DATAPATH/$1/*.m"
else
	for i in $DATAPATH/*; do
		test_all "$i/*.m"
	done
fi
exit $exitcode
