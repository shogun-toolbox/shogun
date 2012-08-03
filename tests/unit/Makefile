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

shogun-lib:
	@$(MAKE) -C $(SHOGUNSRCTOP)/shogun

shogun-unit-test: shogun-lib $(TESTOBJFILES)
	@$(LINK) -o shogun-unit-test $(TESTOBJFILES) $(LINKFLAGS) -L$(SHOGUNSRCTOP)/shogun -lshogun $(LINKFLAGS_GMOCK)

build: shogun-unit-test
	@$(LIBRARY_PATH)=$(SHOGUNSRCTOP)/shogun:$$$(LIBRARY_PATH) ./shogun-unit-test

%$(EXT_OBJ_TEST): %$(EXT_SRC_TEST)
	$(COMP_CPP) -I$(SHOGUNSRCTOP) $(COMPFLAGS_CPP) $(COMPFLAGS_GMOCK_CPP) $(DEFINES) -c $(INCLUDES) -o $@ $<

clean:
	@rm -f ./shogun-unit-test $(TESTOBJFILES)
