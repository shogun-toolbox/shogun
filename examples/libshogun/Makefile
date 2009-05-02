#CXXFLAGS=-I/usr/local/include/shogun
#LINKFLAGS=-L/usr/local/lib -lshogun
LIB_PATH=/usr/lib
CXXFLAGS=-O3 -g $(INCLUDES)
ifneq (,$(LIB_PATH))
	LINKFLAGS+=-L$(LIB_PATH)
endif
LINKFLAGS=-L$(LIB_PATH) -lshogun
TARGETS = rbf_svm gnmpmkl minimal minimal_svm dyn_int rbf_kernel revlin_kernel indirect_object gc_array

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && LD_LIBRARY_PATH="$$LD_LIBRARY_PATH:$(LIB_PATH)" ./$$t >/dev/null; done

%: %.cpp
	gcc $(CXXFLAGS) $< $(LINKFLAGS) -lshogun -o $@
