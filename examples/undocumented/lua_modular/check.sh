#!/bin/bash

status=0

export LUA_PATH=../../../src/interfaces/lua_modular/?.lua\;?.lua
export LUA_CPATH=../../../src/interfaces/lua_modular/?.so

for i in *.lua
do
	echo -n $i
	if lua $i >/dev/null 2>&1
	then
		echo " OK"
	else
		echo " FAIL"
		status=1
	fi
done
exit $status
