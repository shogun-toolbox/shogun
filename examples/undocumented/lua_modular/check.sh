#!/bin/bash

rm -f error.log

export LUA_PATH=../../../src/interfaces/lua_modular/?.lua\;?.lua
export LUA_CPATH=../../../src/interfaces/lua_modular/?.so

for e in *.lua
do
	echo -n $e
	if lua $e >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " ERROR"
		echo "================================================================================" >>error.log
		echo " error in $e ">>error.log
		echo "================================================================================" >>error.log
		lua "$e" >>error.log 2>&1
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
