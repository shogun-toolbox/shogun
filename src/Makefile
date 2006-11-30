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
just-do-it:
#runconfigurehunkstart
	@echo ==================
	@echo SHOGUN - Makefile
	@echo ==================
	@echo
	@echo
	@echo !!! Please run ./configure with the appropriate options first !!!
	@echo
	@echo
	@echo for standalone interface:
	@echo "              ./configure"
	@echo for python interface: 
	@echo "              ./configure --enable-python"
	@echo for python object oriented interface: 
	@echo "              ./configure --enable-python --enable-swig"
	@echo for R interface: 
	@echo "              ./configure --enable-R"
	@echo for octave interface: 
	@echo "              ./configure --enable-octave"
	@echo for matlab interface: 
	@echo "              ./configure --enable-matlab"
	@echo
	@echo For additional options see
	@echo "              ./configure --help"
	@echo
	@exit 1
#runconfigurehunkend
endif

ifeq (.config-local,$(wildcard .config-local))
include .config-local
endif

ifeq (yes,$(USE_SWIG))
TARGET=swig
endif

SRCDIR = .

# SRC/OBJFILES lie all in the subdir of SRCDIR
SVNEntries = $(shell find $(SRCDIR) -name entries -type f)
HEADERFILES	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_HEADER)" )
SRCFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' | grep -v 'configure-')
ifeq (yes,$(USE_SWIG))
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' | grep -v 'configure-' )) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)"| grep -v '.cpuinfo.c' | grep -v 'configure-' )) $(patsubst %.$(EXT_INP_SWIG),%$(EXT_OBJ_SWIG), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
else
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)"| grep -v '.cpuinfo.c' | grep -v 'configure-')) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)"| grep -v '.cpuinfo.c' | grep -v 'configure-'))
endif
SWIGSRCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SRC_SWIG), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
SWIGLIBFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/_\1$(EXT_LIB_SWIG)=')
SWIGPYFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/\1py=')
TAR:=../shogun-$(shell date +%Y-%m-%d).tar

.PHONY: config swig swigclean just-do-it all out clean end condor purif profile dep depend targets warn2err err2warn ctags depgen buildout buildend dist

ifeq (.config,$(wildcard .config))
ifeq (.depend,$(wildcard .depend))
include .depend
just-do-it:	buildout $(TARGET) buildend
else
just-do-it:	depgen .depend
		test -f .depend && $(MAKE)
endif
endif


depgen:	.config
	@echo =================================
	@echo ==== Generating dependencies ====
	@echo =================================

buildout:	.depend .config
	@echo ========================================================================================
	@echo ==== Building shogun for platform $(KERNELNAME)-$(MACHINE) target $(TARGET)
	@echo ==== opts: $(COMPFLAGS_CPP)
	@echo ========================================================================================
	@echo 
buildend:	$(TARGET)	
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

distclean: clean
		rm -f .config tags lib/config.h lib/versionstring.h
swigclean:
	rm -f $(shell find -name "*_wrap.cxx.o" -o -name "*_wrap.cxx" 2>/dev/null) 

clean: swigclean
	rm -f $(shell find . -name "*.cpp.o" -o -name "*.c.o" -o -name "*.so" -o -name "*.dll" -o -name "*~" -o -name ".#*" 2>/dev/null) sg.oct sg.mex* shogun shogun.exe gmon.out .depend configure.log .cpuinfo ./configure-* $(shell find . -type f -name '*.py*' ! -name '__*.py' ! -name '.*.py' | grep -v svn)

dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

ifeq (yes,$(USE_SWIG))
install: $(OBJFILES) $(SRCFILES) $(SWIGSRCFILES) $(SWIGLIBFILES)
	install -d -m755 "$(DESTDIR)$(PYDIR)/shogun"
	install -m755 $(SWIGLIBFILES) $(SWIGPYFILES) "$(DESTDIR)$(PYDIR)/shogun"
	echo -n '__all__= [' > "$(DESTDIR)$(PYDIR)/shogun/__init__.py"
	for i in $(SWIGPYFILES) ; do echo -n $$i | sed -e 's=.*/="=' -e 's/\.py/", /' >>$(DESTDIR)$(PYDIR)/shogun/__init__.py; done
	echo ']' >> "$(DESTDIR)$(PYDIR)/shogun/__init__.py"
else
ifeq (sg.oct,$(TARGET))
install: $(TARGET)
	install -d -m755 $(DESTDIR)$(OCTDIR)
	install -D -m755 $(TARGET) $(DESTDIR)$(OCTDIR)
else
install: $(TARGET)
	install -d -m755 $(DESTDIR)$(BINDIR)
	install -D -m755 $(TARGET) $(DESTDIR)$(BINDIR)
endif
endif

dep:	depend
depend:		.config lib/versionstring.h
	makedepend $(DEFINES) $(INCLUDES) $(shell find $(SRCDIR) -name "*.h" -o -name "*.$(EXT_SRC_CPP)" | grep -v 'configure-' ) -o.$(EXT_OBJ_CPP) -f - >.depend || true
	makedepend $(DEFINES) $(INCLUDES) $(shell find $(SRCDIR) -name "*.h" -o -name "*.$(EXT_SRC_C)" | grep -v '.cpuinfo.c' | grep -v 'configure-' ) -o.$(EXT_OBJ_C) -f - >>.depend || true
	find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -o -name "*.h" -o -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)" | grep -v 'configure-' |  cut -c 3- | $(PYTHON) .generate_link_dependencies.py $(EXT_LIB_SWIG) >>.depend
.depend:	.config lib/versionstring.h
	makedepend $(DEFINES) $(INCLUDES) $(shell find $(SRCDIR) -name "*.h" -o -name "*.$(EXT_SRC_CPP)" | grep -v 'configure-*' ) -o.$(EXT_OBJ_CPP) -f - >.depend || true
	makedepend $(DEFINES) $(INCLUDES) $(shell find $(SRCDIR) -name "*.h" -o -name "*.$(EXT_SRC_C)" | grep -v '.cpuinfo.c' | grep -v 'configure-' ) -o.$(EXT_OBJ_C) -f - >>.depend || true
	find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -o -name "*.h" -o -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)" | grep -v 'configure-' |  cut -c 3- | $(PYTHON) .generate_link_dependencies.py $(EXT_LIB_SWIG) >>.depend

lib/versionstring.h:	$(SVNEntries)
		chmod u+x ./.version.sh
		./.version.sh >lib/versionstring.h

$(OBJFILES): .config .depend lib/versionstring.h 

ifeq (yes,$(USE_SWIG))
$(TARGET): $(OBJFILES) $(SRCFILES) $(SWIGSRCFILES) $(SWIGLIBFILES)
else
$(TARGET): $(OBJFILES) $(SRCFILES)
	$(LINK) $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" | grep -v '.cpuinfo' | grep -v 'configure-' 2>/dev/null) $(LINKFLAGS) -o $@ $(POSTLINKFLAGS)
	#removeconfighunk
endif

_%.${EXT_LIB_SWIG}:	%$(EXT_OBJ_SWIG)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(POSTLINKFLAGS)

%${EXT_OBJ_SWIG}:	%$(EXT_SRC_SWIG)
	$(COMP_CPP) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%${EXT_SRC_SWIG}:	%.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $(DEFINES) $<

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<
