#!/bin/bash

rm -f error.log

if [ -z "${JAVA}" ]
then
	JAVAC=javac
	JAVA=java
fi

export CLASSPATH=/usr/share/java/jblas.jar:../../../src/interfaces/java_modular/shogun.jar:.
export LD_LIBRARY_PATH=../../../src/shogun:../../../src/interfaces/java_modular

${JAVAC} Load.java

for e in $(ls *.java | grep -v Load.java)
do
	echo -n "running $e .."
	if ${JAVAC} $e >/dev/null 2>&1 && ${JAVA} ${e%.java} >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		echo "================================================================================" >>error.log
		echo " error in $e ">>error.log
		echo "================================================================================" >>error.log
		${JAVAC} $e >>error.log 2>&1
		${JAVA} ${e%.java} >>error.log 2>&1
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
