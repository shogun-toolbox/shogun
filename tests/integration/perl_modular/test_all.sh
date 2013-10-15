#!/bin/bash

if test -z "$PERL"
then
    PERL=perl
fi

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
		output=`./test_one.pl "$file"`
		ret=$?

		if [ $ret -eq 1 ] ; then
			echo 'OK'
		else
			echo 'ERROR'
			exitcode=1
			echo $output
		fi
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
exit $exitcode
