#CXXFLAGS=-I/usr/local/include/shogun
#LINKFLAGS=-L/usr/local/lib -lshogun
CXXFLAGS=-O3 -g
LINKFLAGS=-lshogun

all:  rbf_svm gnmpmkl minimal minimal_svm

rbf_svm: rbf_svm.cpp
	gcc rbf_svm.cpp $(LINKFLAGS) -lshogun -o $@

minimal: minimal.cpp
	gcc minimal.cpp $(LINKFLAGS) -lshogun -o $@

minimal_svm: minimal_svm.cpp
	gcc minimal_svm.cpp $(LINKFLAGS) -lshogun -o $@

gnmpmkl: gnmpmkltester.cpp
	gcc gnmpmkltester.cpp $(LINKFLAGS) -lshogun -o $@
