# include config options
#
#
.SUFFIXES: .c .c.o .cpp .cpp.o .h .i .so _wrap.cxx _wrap.cxx.o

ifeq (GNU,$(shell make --version 2>&1 | grep GNU >/dev/null && echo GNU))
MAKE=make
else
MAKE=gmake
endif

all:	just-do-it

ifeq (.config,$(wildcard .config))
include .config
else
just-do-it:	confgen config
		test -f .config && $(MAKE)
endif

ifeq (.config-local,$(wildcard .config-local))
include .config-local
endif

ifeq (yes,$(USE_SWIG))
TARGET=swig
endif

SRCDIR = .

# SRC/OBJFILES lie all in the subdir of SRCDIR
CVSEntries = $(shell find $(SRCDIR)/ -name Entries -type f)
HEADERFILES	= $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_HEADER)" )
SRCFILES 	= $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' )
ifeq (yes,$(USE_SWIG))
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' )) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_C)"| grep -v '.cpuinfo.c' )) $(patsubst %.$(EXT_INP_SWIG),%$(EXT_OBJ_SWIG), $(shell find $(SRCDIR)/ -name "*.$(EXT_INP_SWIG)"))
else
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' )) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR)/ -name "*.$(EXT_SRC_C)"| grep -v '.cpuinfo.c' ))
endif
SWIGSRCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SRC_SWIG), $(shell find $(SRCDIR)/ -name "*.$(EXT_INP_SWIG)" | grep -v 'swigfiles'))
SWIGLIBFILES 	= $(shell find $(SRCDIR)/ -name "*.$(EXT_INP_SWIG)" | grep -v 'swigfiles' | sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/_\1so=')
TAR:=../shogun-$(shell date +%Y-%m-%d).tar

.PHONY: config swig just-do-it all out clean doc texdoc htmldoc end condor purif profile dep depend targets warn2err err2warn ctags confgen depgen buildout buildend dist

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
	@echo ==== Building shogun for platform $(KERNELNAME)-$(MACHINE) target $(TARGET)
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
	@echo ==== Done. Start with shogun.$(UNAME)
	@echo ============================================================
	@echo 

warn2err: $(SRCFILES)
	for i in $^ ; do cat $$i | sed 's/^\#warning/#error/g' >$$i.new ; mv $$i.new $$i ; done
err2warn: $(SRCFILES)
	for i in $^ ; do cat $$i | sed 's/^\#error/#warning/g' >$$i.new ; mv $$i.new $$i ; done

ctags: tags

tags:	$(SRCFILES)
	ctags */*.$(EXT_SRC_CPP) */*.h */*.$(EXT_SRC_C)

linecount:	$(SRCFILES)
	wc -l */*.$(EXT_SRC_CPP) */*.h */*.$(EXT_SRC_C) */*.$(EXT_SRC_SWIG) | grep total

targets:
	@echo "=========="
	@echo "=targets:="
	@echo "=========="
	@egrep '^[a-z]+:' Makefile | grep -v '\.o:' | sed -e 's/:.*//g' -e 's/out//g' -e 's/end//g' -e 's/^/     /g' | sort | uniq

doc:	htmldoc

htmldoc:
	cd ../doc/doxygen/ && doxygen

distclean: clean
		rm -f .config tags lib/config.h lib/versionstring.h

clean: 
	rm -f $(shell find ./ -name "*.cpp.o" -o -name "*.c.o" -o -name "*.so" -o -name "*_wrap.cxx.o" -o -name "*_wrap.cxx" -o -name "*~" -o -name ".#*" 2>/dev/null) sg.mex* sg.so shogun ../doc/doxygen/*.html ../doc/doxygen/shogun.tex gmon.out .depend configure.log .cpuinfo ./configure-* $(shell find ./ -type f -name '*.py*' ! -name '__*.py' ! -name '.*.py' | grep -v svn)

config: .config
.config:	Makefile configure
		chmod u+x ./configure
		./configure #use default options
dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)


dep:	depend
depend:	Makefile .config lib/versionstring.h 
	makedepend $(INCLUDES) $(shell find $(SRCDIR)/ -name "*.h" -o -name "*.$(EXT_SRC_CPP)" ) -o.$(EXT_OBJ_CPP) -f - >.depend
	makedepend $(INCLUDES) $(shell find $(SRCDIR)/ -name "*.h" -o -name "*.$(EXT_SRC_C)" | grep -v '.cpuinfo.c' )  -o.$(EXT_OBJ_C) -f - >>.depend
	makedepend $(INCLUDES) $(shell find $(SRCDIR)/ -name "*.$(EXT_INP_SWIG)" ) -o$(EXT_SRC_SWIG) -f - >>.depend
	find $(SRCDIR)/ -name "*.h" -o -name "*.$(EXT_INP_SWIG)" -o -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)" |  cut -c 3- | $(PYTHON) .generate_link_dependencies.py >>.depend

lib/versionstring.h:	$(CVSEntries)
		chmod u+x ./.version.sh
		./.version.sh >lib/versionstring.h

$(OBJFILES): Makefile .config .depend

ifeq (yes,$(USE_SWIG))
$(TARGET): $(SWIGLIBFILES)
#$(TARGET): $(OBJFILES) $(SRCFILES) $(SWIGSRCFILES)
else
$(TARGET): $(OBJFILES) $(SRCFILES)
	$(LINK) $(shell find $(SRCDIR)/ -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null) $(LINKFLAGS) -o $@ 
	#removeconfighunk
endif

_%.${EXT_LIB_SWIG}:	%$(EXT_OBJ_SWIG)
	$(LINK) $(LINKFLAGS) -o $@ $^

%${EXT_OBJ_SWIG}:	%$(EXT_SRC_SWIG)
	$(COMP_CPP) $(DEFINES) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%${EXT_SRC_SWIG}:	%.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $<

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(DEFINES) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(DEFINES) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<
# DO NOT DELETE
