# include config options
#

ifeq (GNU,$(shell make --version 2>&1 | grep GNU >/dev/null && echo GNU))
MAKE=make
else
MAKE=gmake
endif

all:	just-do-it

ifeq (.config,$(wildcard .config))
include .config
else
#its useless to use $(MAKE) here :-(
just-do-it:	confgen config
		test -f .config && $(MAKE)
endif

ifeq (.config-local,$(wildcard .config-local))
include .config-local
endif

SRCDIR = .

# SRC/OBJFILES lie all in the subdir of SRCDIR
SRCFILES 	= $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' )
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' )) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_C)"| grep -v '.cpuinfo.c' ))

.PHONY: just-do-it all out clean doc texdoc htmldoc end condor purif profile depend targets warn2err err2warn ctags depend confgen depgen buildout buildend

ifeq (.config,$(wildcard .config))
ifeq (.depend,$(wildcard .depend))
include .depend
just-do-it:	buildout $(TARGET) buildend
else
just-do-it:	depgen depend
		test -f .depend && $(MAKE)
endif
endif

confgen:	
	@echo ==================================
	@echo ==== Generating configuration ====
	@echo ==================================
depgen:	
	@echo =================================
	@echo ==== Generating dependencies ====
	@echo =================================

buildout:
	@echo ========================================================================================
	@echo ==== Building genefinder for platform $(KERNELNAME)-$(MACHINE)
	@echo ==== opts: $(COMPFLAGS_CPP)
	@echo ========================================================================================
	@echo 
linkout:
	@echo =================
	@echo ==== Linking ====
	@echo =================
	@echo 
buildend:	
	@echo 
	@echo ============================================================
	@echo ==== Done. Start with genefinder.$(UNAME)
	@echo ============================================================
	@echo 

warn2err: $(SRCFILES)
	for i in $^ ; do cat $$i | sed 's/^\#warning/#error/g' >$$i.new ; mv $$i.new $$i ; done
err2warn: $(SRCFILES)
	for i in $^ ; do cat $$i | sed 's/^\#error/#warning/g' >$$i.new ; mv $$i.new $$i ; done

ctags: tags

tags:	$(SRCFILES)
	ctags */*.cpp */*.h */*.c

targets:
	@echo "=========="
	@echo "=targets:="
	@echo "=========="
	@egrep '^[a-z]+:' Makefile | grep -v '\.o:' | sed -e 's/:.*//g' -e 's/out//g' -e 's/end//g' -e 's/^/     /g' | sort | uniq

doc:	htmldoc

htmldoc:
	cd ../doc/doxygen/ && doxygen

distclean: clean
		rm -f .config tags lib/config.h lib/version.h
clean: .config
	rm -f $(shell find $(SRCDIR)/ -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null) $(TARGET) ../doc/doxygen/*.html ../doc/doxygen/genefinder.tex gmon.out .depend configure.log .cpuinfo ./configure-*

config:	Makefile configure
		chmod u+x ./configure ./.version.sh
		./configure
		./.version.sh `$(COMP_CPP) -dumpversion` >lib/version.h
dep:	depend
depend:	Makefile .config
		chmod u+x ./.version.sh
		./.version.sh `$(COMP_CPP) -dumpversion` >lib/version.h
	makedepend $(INCLUDES) $(shell find $(SRCDIR)/ -name "*.h" -o -name "*.$(EXT_SRC_CPP)" ) -o.$(EXT_OBJ_CPP) -f - >.depend
	makedepend $(INCLUDES) $(shell find $(SRCDIR)/ -name "*.h" -o -name "*.$(EXT_SRC_C)" | grep -v '.cpuinfo.c' )  -o.$(EXT_OBJ_C) -f - >>.depend

$(OBJFILES): Makefile .config .depend

$(TARGET): $(OBJFILES) $(SRCFILES)
	$(LINK) $(shell find $(SRCDIR)/ -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null) $(LINKFLAGS) -o $@ 

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(DEFINES) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<
%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(DEFINES) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<
