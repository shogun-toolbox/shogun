#!/bin/bash

rm -f error.log

if test -z "$1"
then
	for e in *.rb;
	do
		echo -n $e

		if ruby -I../../../src/interfaces/ruby_modular $e >/dev/null 2>&1
		then
			echo " OK"
		else
			echo " ERROR"
			echo "================================================================================" >>error.log
			echo " error in $e ">>error.log
			echo "================================================================================" >>error.log
			ruby -I../../../src/interfaces/ruby_modular "$e" >>error.log 2>&1
			echo "================================================================================" >>error.log
			echo >>error.log
			echo >>error.log
		fi
	done
else
	ruby -I../../../src/interfaces/ruby_modular $1
fi

if test -f error.log 
then
	cat error.log
	exit 1
fi

exit 0
