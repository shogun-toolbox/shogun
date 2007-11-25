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
	@echo "              ./configure --interface=readline"
	@echo for object oriented python interface: 
	@echo "              ./configure --interface=python-modular"
	@echo for python interface: 
	@echo "              ./configure --interface=python"
	@echo for R interface: 
	@echo "              ./configure --interface=R"
	@echo for object oriented R interface: 
	@echo "              ./configure --interface=R-modular"
	@echo for octave interface: 
	@echo "              ./configure --interface=octave"
	@echo for matlab interface: 
	@echo "              ./configure --interface=matlab"
	@echo for library interface:
	@echo "              ./configure --interface=libshogun"
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
SWIGIFILES   	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)")
SWIGSRCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SRC_SWIG), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
ifeq (yes,$(USE_DOXYGEN))
SWIGDOCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SWIG_DOXY), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
else
SWIGDOCFILES 	= 
endif
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

linecount:
	@echo -n "C/C++ source code files (.cpp,.c):"
	@cat $(SRCFILES) | grep -vc '^ *$$'
	@echo -n "C/C++ header files (.h):"
	@cat $(HEADERFILES) | grep -vc '^ *$$'
	@echo -n "SWIG header files (.i):"
	@cat $(SWIGIFILES) | grep -vc '^ *$$'
	@echo -n "Total:"
	@cat $(SRCFILES) $(HEADERFILES) $(SWIGIFILES) | grep -vc '^ *$$'

targets:
	@echo "=========="
	@echo "=targets:="
	@echo "=========="
	@egrep '^[a-z]+:' Makefile | grep -v '\.o:' | sed -e 's/:.*//g' -e 's/out//g' -e 's/end//g' -e 's/^/     /g' | sort | uniq

distclean: clean
		rm -f .config tags lib/config.h lib/versionstring.h
swigclean:
	rm -f $(shell find -name "*_wrap.h" -o -name "*_wrap.cxx.o" -o -name "*_wrap.cxx" 2>/dev/null) 

swigdocclean:
	rm -rf $(shell find -name doxygen_xml -o -name "*_doxygen.i" 2>/dev/null) 

clean: swigclean swigdocclean
	rm -f $(shell find . -name "*.cpp.o" -o -name "*.c.o" -o -name "*.so" -o -name "*.dll" -o -name "*~" -o -name ".#*" 2>/dev/null) sg.oct sg.mex* shogun shogun.exe gmon.out .depend configure.log .cpuinfo ./configure-* $(shell find . -type f -name '*.py*' ! -name '__*.py' ! -name '.*.py' | grep -v svn) $(shell find . -type f -name '*.R')


dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

# installer for swig-python, python, standalone(readline), octave
# matlab and R packages are installed using a different mechanism
#  ( R CMD INSTALL sg and cp :-) 
#
# Python + Swig
ifeq (python-modular,$(INTERFACE))
install: $(OBJFILES) $(SRCFILES) $(SWIGSRCFILES) $(SWIGDOCFILES) $(SWIGLIBFILES)
	install -d -m755 "$(DESTDIR)$(PYDIR)/shogun"
	install -m644 $(SWIGLIBFILES) $(SWIGPYFILES) "$(DESTDIR)$(PYDIR)/shogun"
	echo -n '__all__= [' > "$(DESTDIR)$(PYDIR)/shogun/__init__.py"
	for i in $(SWIGPYFILES) ; do echo -n $$i | sed -e 's=.*/="=' -e 's/\.py/", /' >>$(DESTDIR)$(PYDIR)/shogun/__init__.py; done
	echo ']' >> "$(DESTDIR)$(PYDIR)/shogun/__init__.py"
# Standalone
else
ifeq (cmdline,$(INTERFACE))
install: $(TARGET)
	install -d -m755 $(DESTDIR)$(BINDIR)
	install -m755 $(TARGET) $(DESTDIR)$(BINDIR)
# Octave
else
ifeq (octave,$(INTERFACE))
install: $(TARGET)
	install -d -m755 $(DESTDIR)$(OCTDIR)
	install -m755 $(TARGET) $(DESTDIR)$(OCTDIR)
# Python
else
ifeq (python,$(INTERFACE))
install: $(TARGET)
	install -d -m755 $(DESTDIR)$(PYDIR)
	install -m755 $(TARGET) $(DESTDIR)$(PYDIR)
# R, R-modular and matlab are done separately
else
ifeq (libshogun,$(INTERFACE))
install: $(TARGET)
	install -d -m755 $(DESTDIR)$(LIBDIR)
	install -d -m755 "$(DESTDIR)$(INCDIR)/shogun"
	install -m755 $(TARGET) $(DESTDIR)$(LIBDIR)

	find ./ -mindepth 1 -type d ! -wholename '*.svn*' \
		-exec install -d -m755 "$(DESTDIR)$(INCDIR)/shogun/"\{\} \;
	find ./ -mindepth 1 -type d ! \( -wholename '*.svn*' -o -name '*wrap*' \) \
		-exec sh -c "install -m644 {}/*.h -t $(DESTDIR)$(INCDIR)/shogun/{}" \;
else
install: 
	@echo "The R, R-modular and matlab interface need to be installed"
	@echo "using a different mechanism:"
	@echo
	@echo "R - dyn.load('sg.so')"
	@echo "  - do make -C ../R install"
	@echo
	@echo "matlab - just copy the sg.mex* to a <dir> of your choice"
	@echo "         and addpath <dir>"
	@echo
endif #libshogun
endif #R
endif #python
endif #octave
endif #python+swig

dep:	depend
depend:		.config lib/versionstring.h
	$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.h" -o \
		-name "*.$(EXT_SRC_CPP)" | grep -v 'configure-' ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_CPP):/g' >.depend || true
	$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.h" -o \
		-name "*.$(EXT_SRC_C)" | grep -v 'configure-' ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_C):/g' >>.depend || true
	find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -o \
		-name "*.h" -o -name "*.$(EXT_SRC_CPP)" -o -name "*.$(EXT_SRC_CPP)" | \
		grep -v 'configure-' |  \
		cut -c 3- | \
		$(PYTHON) .generate_link_dependencies.py $(EXT_LIB_SWIG) >>.depend
.depend:	.config lib/versionstring.h
	$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.h" -o \
		-name "*.$(EXT_SRC_CPP)" | grep -v 'configure-' ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_CPP):/g' >.depend || true
	$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.h" -o \
		-name "*.$(EXT_SRC_C)" | grep -v 'configure-' ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_C):/g' >>.depend || true
	find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -o \
		-name "*.h" -o -name "*.$(EXT_SRC_CPP)" -o -name "*.$(EXT_SRC_CPP)" | \
		grep -v 'configure-' |  \
		cut -c 3- | \
		$(PYTHON) .generate_link_dependencies.py $(EXT_LIB_SWIG) >>.depend

lib/versionstring.h:	$(SVNEntries)
		sh .version.sh > $@
tests:
		cd ../$(INTERFACE)/testcases && ./run.sh	

ChangeLog:
		@svn2cl --group-by-day --authors=.authors 2>/dev/null || true

$(OBJFILES): .config .depend lib/versionstring.h ChangeLog

ifeq (yes,$(USE_SWIG))
$(TARGET): $(OBJFILES) $(SRCFILES) $(SWIGSRCFILES) $(SWIGDOCFILES) $(SWIGLIBFILES) 
else
$(TARGET): $(OBJFILES) $(SRCFILES)
	$(LINK) $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" | grep -v '.cpuinfo' | grep -v 'configure-' 2>/dev/null) $(LINKFLAGS) -o $@ $(POSTLINKFLAGS)
	#removeconfighunk
endif

_%.${EXT_LIB_SWIG}:	%$(EXT_OBJ_SWIG)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(POSTLINKFLAGS)

%${EXT_OBJ_SWIG}:	%$(EXT_SRC_SWIG)
	$(COMP_CPP) $(COMPFLAGS_SWIG_CPP) -c $(INCLUDES) -o $@ $<

ifeq (yes,$(USE_DOXYGEN))
%${EXT_SRC_SWIG}:	%$(EXT_SWIG_DOXY) %.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $(DEFINES) $*.$(EXT_INP_SWIG)

%$(EXT_SWIG_DOXY): %$(EXT_DOXY)
	$(DOXYGEN) $<
	$(PYTHON) .doxy2swig.py $(@D)/doxygen_xml/index.xml $@
else
%${EXT_SRC_SWIG}: %.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $(DEFINES) $*.$(EXT_INP_SWIG)
endif

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<

