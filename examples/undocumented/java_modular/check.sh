#!/bin/bash
if [ -z "${JAVA}" ]
then
	JAVAC=javac
	JAVA=java
fi

export CLASSPATH=/usr/share/java/jblas.jar:../../../src/java_modular/shogun.jar:.
${JAVAC} Load.java
${JAVAC} $(ls *.java | sed 's/Load.java//')

export LD_LIBRARY_PATH=../../../src/libshogun:../../../src/java_modular
for filename in *.class
  do
  [ "$filename" = "Load.class" ] && continue
  echo "running ${filename%.class} .."
  ${JAVA} ${filename%.class}
done
