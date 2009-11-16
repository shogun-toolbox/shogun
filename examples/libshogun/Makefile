LIB_PATH=/usr/lib
INC_PATH=/usr/include

INCLUDES=
LIBS=-lshogun -lblas

LINKFLAGS=-L$(LIB_PATH) $(LIBS)
CXXFLAGS=-O3 -g -I$(INC_PATH) $(INCLUDES)

LIBRARY_PATH=LD_LIBRARY_PATH

CC=c++

TARGETS = basic_minimal classifier_libsvm classifier_minimal_svm classifier_mklmulticlass \
	kernel_gaussian kernel_revlin library_dyn_int library_gc_array library_indirect_object

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && $(LIBRARY_PATH)="$$$(LIBRARY_PATH):$(LIB_PATH)" ./$$t >/dev/null; done

%: %.cpp
	$(CC) $(CXXFLAGS) $< $(LINKFLAGS) -o $@
