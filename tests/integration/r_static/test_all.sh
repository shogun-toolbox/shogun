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
		if grep -q $file ../blacklist
		then
			echo 'SKIPPING'
		else
			dump=`echo ${file} | grep WDSVMOcas`
			if [ $? -eq 0 ]; then
				echo WDSVMOcas totally unsupported: ERROR
			else
				echo -n "${file}"
				echo -n -e "\t\t"

				output=`R --no-save --no-restore --no-readline --slave ${file} < test_one.R`
				if [ $? -ne 0 ]; then
					exitcode=1
					echo ERROR
					echo ${output}
				else
					echo OK
				fi
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
