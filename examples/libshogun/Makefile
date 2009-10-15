#CXXFLAGS=-I/usr/local/include/shogun
#LINKFLAGS=-L/usr/local/lib -lshogun

LIBRARY_PATH=LD_LIBRARY_PATH
LIB_PATH=/usr/lib
CXXFLAGS=-O3 -g $(INCLUDES)
ifneq (,$(LIB_PATH))
	LINKFLAGS+=-L$(LIB_PATH)
endif
LINKFLAGS=-L$(LIB_PATH) -lshogun
TARGETS = rbf_svm mklmulticlass minimal minimal_svm dyn_int rbf_kernel revlin_kernel indirect_object gc_array
CC=c++

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && $(LIBRARY_PATH)="$$$(LIBRARY_PATH):$(LIB_PATH)" ./$$t >/dev/null; done

%: %.cpp
	$(CC) $(CXXFLAGS) $< $(LINKFLAGS) -lshogun -o $@
