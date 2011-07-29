#!/bin/bash
if [ -z "${MONO}" ]
then
	MONOC=gmcs
	MONO=mono
fi

args=$#
if [ $args -gt 1 ]; then
echo "Usage: ./check.sh   check all examples.
        ./check.sh name.cs  check one example."
exit 0
fi

export LD_LIBRARY_PATH=../../../src/shogun:../../../src/interfaces/csharp_modular

if [ $args -eq 0 ]; then
  #${MONOC} Load.cs
  ${MONOC} $(ls *.cs | sed 's/Load.cs//')

  for filename in $(ls *.cs | sed 's/Load.cs//')
  do
	  ${MONOC} $filename ../../../src/interfaces/csharp_modular/*.cs
   echo "running ${filename%.cs}.exe .."
   ${MONO} ${filename%.class}.exe >/dev/null
  done
fi

if [ $args -eq 1 ]; then
  arg=("$@")
  #${MONOC} Load.cs
  ${MONOC} ${arg[0]} ../../../src/interfaces/csharp_modular/*.cs
  ${MONO} ${arg[0]%.cs}.exe >/dev/null
fi
