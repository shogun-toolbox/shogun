#CXXFLAGS=-I/usr/local/include/shogun
#LINKFLAGS=-L/usr/local/lib -lshogun
CXXFLAGS=-O3 -g
LINKFLAGS=-lshogun

all:  rbf_svm gnmpmkl

rbf_svm: rbf_svm.cpp
	gcc rbf_svm.cpp $(LINKFLAGS) -lshogun -o $@

gnmpmkl: gnmpmkltester.cpp
	gcc gnmpmkltester.cpp $(LINKFLAGS) -lshogun -o $@
