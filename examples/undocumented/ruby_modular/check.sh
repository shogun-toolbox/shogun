#!/bin/bash

rm -f error.log

if test -z "$1"
then
	for e in $(ls -1 *.rb | grep -v ruby | grep -v shogun_helpers)
	do
		echo -n $e

		if ruby $e >/dev/null 2>&1
		then
			echo " OK"
		else
			echo " ERROR"
			echo "================================================================================" >>error.log
			echo " error in $e ">>error.log
			echo "================================================================================" >>error.log
			ruby "$e" >>error.log 2>&1
			echo "================================================================================" >>error.log
			echo >>error.log
			echo >>error.log
		fi
	done
else
	ruby $1
fi

if test -f error.log 
then
	cat error.log
	exit 1
fi

exit 0
