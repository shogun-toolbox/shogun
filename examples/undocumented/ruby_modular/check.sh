#!/bin/bash

for i in *.rb;
do
	ruby -I../../../src/interfaces/ruby_modular $i
done
