SHOGUNSRCTOP=../../src

ifeq ($(SHOGUNSRCTOP)/.config,$(wildcard $(SHOGUNSRCTOP)/.config))
CONFIGPATH = $(SHOGUNSRCTOP)
all: build
include $(SHOGUNSRCTOP)/.config
else
all:
	@echo !!! Please run ./configure in $(SHOGUNSRCTOP) directory with the appropriate options first !!!
endif

TESTSRCFILES = $(shell find $(SRCDIR) -name "*$(EXT_SRC_TEST)")
TESTOBJFILES = $(patsubst %$(EXT_SRC_TEST),%$(EXT_OBJ_TEST), $(TESTSRCFILES))

gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc
	$(COMP_CPP) $(COMPFLAGS_CPP) $(INCLUDES) $(DEFINES) -I$(SHOGUNSRCTOP) $(COMPFLAGS_GTEST_CPP) $(COMPFLAGS_GMOCK_CPP) -I$(GTEST_DIR) -I$(GMOCK_DIR) -c $<

gmock-all.o: $(GMOCK_DIR)/src/gmock-all.cc
	$(COMP_CPP) $(COMPFLAGS_CPP) $(INCLUDES) $(DEFINES) -I$(SHOGUNSRCTOP) $(COMPFLAGS_GTEST_CPP) $(COMPFLAGS_GMOCK_CPP) -I$(GTEST_DIR) -I$(GMOCK_DIR) -c $<

libgmock.a: gtest-all.o gmock-all.o
	ar -rv libgmock.a gtest-all.o gmock-all.o 

shogun-lib:
	@$(MAKE) -C $(SHOGUNSRCTOP)/shogun

shogun-unit-test: libgmock.a shogun-lib $(TESTOBJFILES)
	@$(LINK) -o shogun-unit-test $(TESTOBJFILES) $(LINKFLAGS) -L$(SHOGUNSRCTOP)/shogun -lshogun -lpthread libgmock.a $(LINKFLAGS_GMOCK)

build: shogun-unit-test
	@$(LIBRARY_PATH)=$(SHOGUNSRCTOP)/shogun:$$$(LIBRARY_PATH) ./shogun-unit-test

%$(EXT_OBJ_TEST): %$(EXT_SRC_TEST)
	$(COMP_CPP) -I$(SHOGUNSRCTOP) $(COMPFLAGS_CPP) $(COMPFLAGS_GTEST_CPP) $(COMPFLAGS_GMOCK_CPP) $(DEFINES) -c $(INCLUDES) -o $@ $<

clean:
	@rm -f ./shogun-unit-test libgmock.a gmock-all.o gtest-all.o $(TESTOBJFILES)
