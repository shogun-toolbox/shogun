# This sed script patches the Makefile for the R build.

s/#removeconfighunk/rm -f .config/
/#runconfigurehunkstart/,/#runconfigurehunkend/c\	./configure --interface=R && make\n\texit 0
