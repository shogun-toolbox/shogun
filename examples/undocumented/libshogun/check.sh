#!/bin/bash

rm -f error.log

MAKEOPTS=""
test -n "$LIBRARY_PATH" && MAKEOPTS="LIBRARY_PATH=${LIBRARY_PATH} ${MAKEOPTS}"
test -n "$LIB_PATH" && MAKEOPTS="LIB_PATH=${LIB_PATH} ${MAKEOPTS}"
test -n "$INC_PATH" && MAKEOPTS="INC_PATH=${INC_PATH} ${MAKEOPTS}"
test -n "$LIBS" && MAKEOPTS="LIBS=${LIBS} ${MAKEOPTS}"

for e in `make print_targets`
do
	echo -n "running $e .."
	if make $MAKEOPTS "$e" >/dev/null 2>&1 && "./$e" >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		echo "================================================================================" >>error.log
		echo " error in $e ">>error.log
		echo "================================================================================" >>error.log
		make $MAKEOPTS "$e" >>error.log 2>&1 && "./$e">>error.log 2>&1
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
