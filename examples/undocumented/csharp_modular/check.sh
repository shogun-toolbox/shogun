#!/usr/bin/env bash

rm -f error.log

if [ -z "${MONO}" ]
then
	MONOC=gmcs
	MONO=mono
fi

export LD_LIBRARY_PATH=../../../src/shogun:../../../src/interfaces/csharp_modular
export MONO_PATH=../../../src/interfaces/csharp_modular

MONOFLAGS="Load.cs /lib:../../../src/interfaces/csharp_modular /r:modshogun"


FILES=$@

test -z "$FILES" && FILES=$(ls *.cs | grep -v Load.cs )

for e in $FILES
do
	echo -n "running $e .."
	if ${MONOC} $e $MONOFLAGS >/dev/null 2>&1 && \
		${MONO} ${e%.cs}.exe >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		echo "================================================================================" >>error.log
		echo " error in $e ">>error.log
		echo "================================================================================" >>error.log
		${MONOC} $e $MONOFLAGS >>error.log 2>&1 && \
		${MONO} ${e%.cs}.exe >>error.log 2>&1
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
