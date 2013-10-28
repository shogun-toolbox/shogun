#!/usr/bin/env bash

DATAPATH='../data'
exitcode=0

function test_all () {
	datapath=${1}

	if echo "$datapath" | grep -q '\.\./data/tests'
	then
		continue
	fi
	echo "*** Testing in ${datapath}"
	sleep 1
	for file in ${datapath}; do
		echo -n "${file}"
		echo -n -e "\t\t"

		if grep -q $file ../blacklist
		then
			echo 'SKIPPING'
		else
			output=`./test_one.sh ${file}`
			ans=`echo ${output} | grep 'ans =' | awk '{print $NF}'`
			if [ -z ${ans} ]; then
				ans=0
			fi

			# thanks to matlab, 1 means ok and 0 means error
			if [ ${ans} -eq 0 ]; then
				exitcode=1
				echo ERROR
				# remove octave banner
				echo ${output} | grep -v 'GNU Octave'
			else
				echo OK
			fi
		fi
	done
	sleep 1
	echo
}

if [ -n "${1}" ]; then
	test_all "${DATAPATH}/${1}/*.m"
else
	for i in ${DATAPATH}/*; do
		test_all "${i}/*.m"
	done
fi
exit $exitcode
