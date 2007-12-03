# !/bin/bash

DATAPATH='../../testsuite/data'

function test_all () {
	datapath="$1"
	echo "*** Testing in $datapath"
	sleep 1
	for file in $datapath; do
		echo -n "$file"
		echo -n -e "\t\t"

		python test.py "$file" > /dev/null
		ret=$?

		if [ $ret -eq 0 ] ; then
			echo 'OK'
		else
			echo 'ERROR'
		fi
	done
	sleep 1
	echo
}

if [ -n "$1" ]; then
	test_all "$DATAPATH/$1/*.m"
else
	test_all "$DATAPATH/kernel/*.m"
	test_all "$DATAPATH/distance/*.m"
	test_all "$DATAPATH/svm/*.m"
fi
