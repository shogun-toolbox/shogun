#!/bin/bash

DATAPATH='../data'

function test_all () {
	dir=${1}
	echo "*** Testing in ${dir}"
	sleep 1
	for file in ${dir}; do
		dump=`echo ${file} | grep WDSVMOcas`
		if [ $? -eq 0 ]; then
			echo WDSVMOcas totally unsupported: ERROR
		else
			echo -n "${file}"
			echo -n -e "\t\t"

			output=`R --no-save --no-restore --no-readline --slave ${file} < test_one.R`
			if [ $? -ne 0 ]; then
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


if [ -n "${1}" ]; then
	test_all "${DATAPATH}/${1}/*.m"
else
	for i in ${DATAPATH}/*; do
		test_all "${i}/*.m"
	done
fi
