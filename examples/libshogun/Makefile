#CXXFLAGS=-I/usr/local/include/shogun
#LINKFLAGS=-L/usr/local/lib -lshogun
CXXFLAGS=-O3 -g
LINKFLAGS=-lshogun
TARGETS = rbf_svm gnmpmkl minimal minimal_svm dyn_int rbf_kernel revlin_kernel indirect_object

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && ./$$t >/dev/null; done

%: %.cpp
	gcc $(CXXFLAGS) $< $(LINKFLAGS) -lshogun -o $@
