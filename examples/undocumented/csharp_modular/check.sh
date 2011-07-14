#!/bin/bash
if [ -z "${JAVA}" ]
then
	JAVAC=javac
	JAVA=java
fi

args=$#
if [ $args -gt 1 ]; then
echo "Usage: ./check.sh   check all examples.
        ./check.sh name.cs  check one example."
exit 0
fi

export CLASSPATH=/usr/share/java/jblas.jar:../../../src/java_modular/shogun.jar:.
export LD_LIBRARY_PATH=../../../src/libshogun:../../../src/java_modular

if [ $args -eq 0 ]; then
  ${JAVAC} Load.cs
  ${JAVAC} $(ls *.cs | sed 's/Load.cs//')

  for filename in *.class
  do
   [ "$filename" = "Load.class" ] && continue
   echo "running ${filename%.class} .."
   ${JAVA} ${filename%.class} >/dev/null
  done
fi

if [ $args -eq 1 ]; then
  arg=("$@")
  ${JAVAC} Load.cs
  ${JAVAC} ${arg[0]}
  ${JAVA} ${arg[0]%.cs} >/dev/null
fi

