#!/bin/bash
if [ -z "${JAVA}" ]
then
	JAVAC=javac
	JAVA=java
fi

export CLASSPATH=/usr/share/java/jblas.jar:../../../src/java_modular/shogun.jar:.
${JAVAC} Load.java
${JAVAC} *.java

export LD_LIBRARY_PATH=../../../src/libshogun:../../../src/java_modular
for filename in *.class
  do
  echo "running ${filename%.class} .."
  ${JAVA} ${filename%.class}
done
