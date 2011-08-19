#!/bin/bash

rm -f error.log

for e in *.R
do
	echo -n "running $e .."
	if R --slave -f "$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		echo "================================================================================" >>error.log
		echo " error in $e ">>error.log
		echo "================================================================================" >>error.log
		R --slave -f "$e" >>error.log 2>&1
		echo "================================================================================" >>error.log
		echo >>error.log
		echo >>error.log
	fi
done

test -f error.log && ( cat error.log ; exit 1 )
exit 0
