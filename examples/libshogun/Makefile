#CXXFLAGS=-I/usr/local/include/shogun
#LINKFLAGS=-L/usr/local/lib -lshogun

LIBRARY_PATH=LD_LIBRARY_PATH
LIB_PATH=/usr/lib
CXXFLAGS=-O3 -g $(INCLUDES)
ifneq (,$(LIB_PATH))
	LINKFLAGS+=-L$(LIB_PATH)
endif
LINKFLAGS=-L$(LIB_PATH) -lshogun
TARGETS = basic_minimal classifier_libsvm classifier_minimal_svm classifier_mklmulticlass \
	kernel_gaussian kernel_revlin library_dyn_int library_gc_array library_indirect_object
CC=c++

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && $(LIBRARY_PATH)="$$$(LIBRARY_PATH):$(LIB_PATH)" ./$$t >/dev/null; done

%: %.cpp
	$(CC) $(CXXFLAGS) $< $(LINKFLAGS) -lshogun -o $@
