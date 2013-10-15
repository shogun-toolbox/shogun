#!/usr/bin/env bash

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

if test -f error.log
then
	cat error.log
	exit 1
fi

exit 0
