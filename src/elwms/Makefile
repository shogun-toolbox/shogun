# note the word Template (all lowercase and all uppercase are replaced with the current interface name)
#
.SUFFIXES: .c .c.o .cpp .cpp.o .h .i .so .a _wrap.cxx _wrap.cxx.o
.PHONY: all out clean end targets warn2err err2warn ctags buildout buildend dist

SRCDIR = .

ifneq (../.config,$(wildcard ../.config))
all:
	@echo !!! Please run ./configure in the top level directory with the appropriate options first !!!
else
include ../.config
ifeq (../.config-local,$(wildcard ../.config-local))
include ../.config-local
endif

ifeq (libshogun,$(findstring libshogun,$(ELWMS_TARGET)))
	LIBNAME=$(shell echo $(ELWMS_TARGET) | cut -f 1 -d '.')
	SONAME=$(ELWMS_SONAME)
	OTHER_TARGETS=$(LIBNAME).a
	LIBSYMLINKCMD=rm -f $(LIBNAME).$(EXT_LIB) $(SONAME); ln -s $(ELWMS_TARGET) $(SONAME) ; ln -s $(ELWMS_TARGET) $(LIBNAME).$(EXT_LIB)

ifeq (libshogun.,$(findstring libshogun.,$(ELWMS_TARGET)))
DEPDEPS=lib/versionstring.h
endif

else

ifeq (elwms,$(findstring elwms,$(ELWMS_TARGET)))
	LIBSYMLINKCMD=rm -f elwms.oct elwms.mex*; ln -s $(ELWMS_TARGET) elwms.oct; \
				  if test -n "$(MATLAB_TARGET)";\
				  then \
				    ln -s $(ELWMS_TARGET) \
				      elwms.$(shell echo $(MATLAB_TARGET) | cut -f 2 -d '.');\
				  fi
endif

ifeq (modular,$(findstring modular,$(ELWMS_TARGET)))
INCLUDES += $(INCLUDES_MODULAR_INTERFACES)
LINKFLAGS += $(LINKFLAGS_MODULAR_INTERFACES)
SWIGFLAGS += $(SWIGFLAGS_ELWMS) $(INCLUDES_MODULAR_INTERFACES) -I.. -I.
OTHERDEPS = $(SWIGSRCFILES) $(SWIGLIBFILES) 
ifeq (yes,$(USE_DOXYGEN))
OTHERDEPS += $(SWIGDOCFILES)
endif
$(shell for i in ../modular/*.* ; do test -f `basename $$i` || ln -s $$i . ; done )
else
	INCLUDES += $(INCLUDES_STATIC_INTERFACES)
	LINKFLAGS += $(LINKFLAGS_STATIC_INTERFACES)
endif

ifeq (r_modular,$(ELWMS_TARGET))
OTHERDEPS += $(SWIGRFILES) $(SWIGRDATAFILES)
endif

endif

all: $(ELWMS_TARGET) $(OTHER_TARGETS) $(OTHERDEPS)
endif

INCLUDES += $(INCLUDES_ELWMS)
DEFINES += $(DEFINES_ELWMS)
COMPFLAGS_CPP += $(DEFINES)
COMPFLAGS_C += $(DEFINES)
LINKFLAGS += $(LINKFLAGS_ELWMS)
POSTLINKFLAGS += $(POSTLINKFLAGS_ELWMS)

# SRC/OBJFILES lie all in the subdir of SRCDIR
SVNEntries = $(shell find $(SRCDIR) -name entries -type f)
HEADERFILES	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_HEADER)" )
SRCFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)")

ifeq (modular,$(findstring modular,$(ELWMS_TARGET)))
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)" )) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" )) $(patsubst %.$(EXT_INP_SWIG),%$(EXT_OBJ_SWIG), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
else
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)")) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)"))
endif

SWIGIFILES   	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)")
SWIGSRCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SRC_SWIG), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
SWIGDOCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SWIG_DOXY), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
SWIGLIBFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/$(PRE_LIB_SWIG_ELWMS)\1$(EXT_LIB_SWIG_ELWMS)=' | sort)
SWIGPYFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/\1py=')
SWIGRFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/\1R=')
SWIGRDATAFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/\1RData=')
TAR:=../shogun-$(shell date +%Y-%m-%d).tar

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
		@rm -f tags .depend

swigclean:
	@rm -f $(shell find -name "*_wrap.h" -o -name "*_wrap.cxx.o" -o -name "*_wrap.cxx" 2>/dev/null) 

swigdocclean:
	@rm -rf $(shell find -name doxygen_xml -o -name "*_doxygen.i" 2>/dev/null) 


clean:
	@rm -f $(shell find . -name "*.cpp.o" -o -name "*.c.o")
	@rm -f $(shell find . -name "*.a" -o -name "*.so*" -o -name "*.dylib" -o -name "*.dll" -o -name "*~" -o -name ".#*")
	@rm -f $(shell find . -type f -name '*.py' ! -name '__*.py' ! -name '.*.py' | grep -v svn)
	@rm -f $(shell find . -type f -name '*.pyc' -o -name '*.R')
	@rm -f $(shell find . -type f -name '*.RData')
	@rm -f $(shell find . -type f -name '*.oct')
	@rm -f $(shell find . -type l -name '*.i')
	@rm -f $(shell for i in `find . -type l -name '*.doxy'` ; do rm -rf `basename $$i .doxy` ; done)
	@rm -f $(shell find . -type l -name '*.doxy')
	@rm -f $(shell find . -type f -name '*doxygen.i')
	@rm -f $(shell find . -type f -name '*wrap.cxx*')
	@rm -f $(shell find . -type f -name '*.swp')
	@rm -f $(shell find . -type f -name 'octave-core')
	@rm -f sg.mex* shogun shogun.exe gmon.out .depend configure.log .cpuinfo ./configure-* cplex.log
	@rm -f elwms.mex* elwms.so elwms.dylib elwms.dll elwms.oct
	@rm -f $(ELWMS_TARGET)

install: install-elwms

install-cmdline: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(BINDIR)
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(BINDIR)

install-libshogunui: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(LIBDIR)
	install -d -m755 "$(DESTDIR)$(INCDIR)/shogun/ui"
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(LIBDIR)
	install -m755 libshogunui.a $(DESTDIR)$(LIBDIR)
	install -m644 *.h "$(DESTDIR)$(INCDIR)/shogun/ui"
	find "$(DESTDIR)$(INCDIR)/shogun/ui" -name '*.h' -exec $(SEDMI) 's/#include "\([^"]*\)"/#include <shogun\/ui\/\1>/' {} \;
	rm -f $(DESTDIR)$(LIBDIR)/libshogunui.$(EXT_LIB) $(DESTDIR)$(LIBDIR)/$(ELWMS_SONAME)
	cd $(DESTDIR)$(LIBDIR) && ( ln -s $(ELWMS_TARGET) libshogunui.$(EXT_LIB); \
	ln -s $(ELWMS_TARGET) $(ELWMS_SONAME) )
	-ldconfig

install-libshogun: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(LIBDIR)
	install -d -m755 "$(DESTDIR)$(INCDIR)/shogun"
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(LIBDIR)
	install -m755 libshogun.a $(DESTDIR)$(LIBDIR)

	find ./ -mindepth 1 -type d ! \( -path '*.svn*' -o -name '*interface*' -o -name '*guilib*' -o -name '*doxygen*' \)\
		-exec install -d -m755 "$(DESTDIR)$(INCDIR)/shogun/"\{\} \;
	find ./ -mindepth 1 -type d ! \( -path '*.svn*' -o -name '*wrap*' -o -name '*interface*' -o -name '*guilib*' -o -name '*doxygen*' \) \
		-exec sh -c "install -m644 {}/*.h $(DESTDIR)$(INCDIR)/shogun/{}" \;
	find "$(DESTDIR)$(INCDIR)/shogun/" -name '*.h' -exec $(SEDMI) 's/#include "\([^"]*\)"/#include <shogun\/\1>/' {} \;
	rm -f $(DESTDIR)$(LIBDIR)/libshogun.$(EXT_LIB) $(DESTDIR)$(LIBDIR)/$(ELWMS_SONAME)
	cd $(DESTDIR)$(LIBDIR) && ( ln -s $(ELWMS_TARGET) libshogun.$(EXT_LIB); \
	ln -s $(ELWMS_TARGET) $(ELWMS_SONAME) )
	-ldconfig

install-python: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(PYDIR)
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(PYDIR)

install-octave: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(OCTDIR)
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(OCTDIR)

install-r: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(RDIR)/sg/demo
	install -d -m755 $(DESTDIR)$(RDIR)/sg/libs
	install -d -m755 $(DESTDIR)$(RDIR)/sg/Meta
	install -d -m755 $(DESTDIR)$(RDIR)/sg/R
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(RDIR)/sg/libs
	install -m644 ../../examples/r/*.R ../../examples/r/graphical/*.R $(DESTDIR)$(RDIR)/sg/demo
	../.r-install.sh $(DESTDIR)$(RDIR) sg

install-elwms: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(PYDIR)
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(PYDIR)
	install -d -m755 $(DESTDIR)$(OCTDIR)
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(OCTDIR)
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/demo
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/libs
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/Meta
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/R
	install -m755 $(ELWMS_TARGET) $(DESTDIR)$(RDIR)/elwms/libs
	install -m644 ../../examples/r/*.R ../../examples/r/graphical/*.R $(DESTDIR)$(RDIR)/elwms/demo
	../.r-install.sh $(DESTDIR)$(RDIR) elwms

install-matlab: $(ELWMS_TARGET)
	@echo "The matlab interface needs to be installed"
	@echo "using a different mechanism:"
	@echo
	@echo " just copy the sg.mex* to a <dir> of your choice"
	@echo " and addpath <dir>"
	@echo

install-python_modular: $(ELWMS_TARGET)
	install -d -m755 "$(DESTDIR)$(PYDIR)/shogun"
	install -m644 $(SWIGLIBFILES) $(SWIGPYFILES) "$(DESTDIR)$(PYDIR)/shogun"
	echo '__all__= [' > "$(DESTDIR)$(PYDIR)/shogun/__init__.py"
	for i in $(SWIGPYFILES) ; do echo $$i | sed -e 's=.*/="=' -e 's/\.py/", /' >>$(DESTDIR)$(PYDIR)/shogun/__init__.py; done
	echo ']' >> "$(DESTDIR)$(PYDIR)/shogun/__init__.py"

install-r_modular: $(ELWMS_TARGET)
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/demo
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/libs
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/Meta
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/R
	install -m644 ../../examples/r_modular/*.R $(DESTDIR)$(RDIR)/shogun/demo
	install -m755 *.so $(DESTDIR)$(RDIR)/shogun/libs
	install -m755 *.RData $(DESTDIR)$(RDIR)/shogun/R
	../.r-install.sh $(DESTDIR)$(RDIR) shogun

install-octave_modular: $(ELWMS_TARGET)
	install -d -m755 "$(DESTDIR)$(OCTDIR)/shogun"
	install -m755 $(SWIGLIBFILES) "$(DESTDIR)$(OCTDIR)/shogun"
	echo '%Initialize shogun by loading all of its modules' > "$(DESTDIR)$(OCTDIR)/shogun/init_shogun.m"
	for i in $(SWIGLIBFILES) ; do echo $$i | sed -e 's=.*/==' -e 's/\.oct/;/' >>$(DESTDIR)$(OCTDIR)/shogun/init_shogun.m; done

run-testsuite: run-testsuite-elwms
check-examples: check-examples-elwms

run-testsuite-libshogun:
	true
check-examples-libshogun:
	cd ../../examples/libshogun && make check-examples

run-testsuite-libshogunui:
	true
check-examples-libshogunui:
	true

run-testsuite-elwms:
	true
check-examples-elwms:
	true

run-testsuite-python:
	cd ../../testsuite/python && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./test_all.sh	)
check-examples-python:
	cd ../../examples/python && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./check.sh	)

run-testsuite-r:
	cd ../../testsuite/r && ( R_LIBS="$(DESTDIR)$(RDIR)" ./test_all.sh )
check-examples-r:
	cd ../../examples/r && ( R_LIBS="$(DESTDIR)$(RDIR)" ./check.sh )

run-testsuite-cmdline:
	cd ../../testsuite/cmdline && ( PATH="$(DESTDIR)$(BINDIR)"  ./test_all.sh )
check-examples-cmdline:
	cd ../../examples/cmdline && ( PATH="$(DESTDIR)$(BINDIR)"  ./check.sh )

run-testsuite-octave:
	cd ../../testsuite/octave && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR):" ./test_all.sh - octave )
check-examples-octave:
	cd ../../examples/octave && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR):" ./check.sh )

run-testsuite-matlab:
	cd ../../testsuite/matlab && ( ln -sf ../../src/matlab/$(ELWMS_TARGET) ; ./test_all.sh - matlab)
check-examples-matlab:
	cd ../../examples/matlab && ./check.sh

run-testsuite-python_modular:
	cd ../../testsuite/python_modular && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./test_all.sh    )
check-examples-python_modular:
	cd ../../examples/python_modular && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./check.sh    )

run-testsuite-r_modular:
	cd ../../testsuite/r_modular && ( R_LIBS="$(DESTDIR)$(RDIR)" ./test_all.sh )
check-examples-r_modular:
	cd ../../examples/r_modular && ( R_LIBS="$(DESTDIR)$(RDIR)" ./check.sh )

run-testsuite-octave_modular:
	cd ../../testsuite/octave_modular && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)/shogun:" ./test_all.sh )
check-examples-octave_modular:
	cd ../../examples/octave_modular && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)/shogun:" ./check.sh )

lib/versionstring.h:    $(SVNEntries) 
	        sh ../.version.sh > $@

doc-libshogun:	lib/versionstring.h
		-@$(SEDMI) "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = `sed -n 's/^#define VERSION_RELEASE "\(.*\)_"/\1/p' $@`/" ../../doc/Doxyfile
		cd ../../doc && make

doc-libshogunui:
	true

doc-elwms:
	true

doc-octave:
	echo "/** \page staticoctave Static Matlab/Octave Interface Function Reference" >x
	-echo "sg('set_output', 'y');sg('help','doxygen');" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)" octave )
	echo "*/" >>y
	cat x y >../../doc/pages/StaticOctaveInterface.mainpage
	rm -f x y

doc-cmdline:
	echo "/** \page staticcmdline Static Command Line Interface Function Reference" >x
	-bash -c "echo -e \"set_output y\nhelp doxygen\n\"" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun  PATH="./:$(DESTDIR)$(BINDIR)" shogun )
	echo "*/" >>y
	cat x y >../../doc/pages/StaticCommandLineInterface.mainpage
	rm -f x y

doc-python:
	echo "/** \page staticpython Static Python Interface Function Reference" >x
	-bash -c "echo -e \"from sg import sg\nsg('set_output', 'y')\nsg('help','doxygen')\n\"" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun  PYTHONPATH="$(DESTDIR)$(PYDIR)" python )
	echo "*/" >>y
	cat x y >../../doc/pages/StaticPythonInterface.mainpage
	rm -f x y

doc-r:
	echo "/** \page staticr Static R Interface Function Reference" >x
	-echo "library(sg);sg('set_output', 'y');sg('help','doxygen');" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun  R_PATH="$(DESTDIR)$(RDIR)" R --no-save)
	echo "*/" >>y
	cat x y >../../doc/pages/StaticRInterface.mainpage
	rm -f x y

doc-matlab:
	@echo "Check the octave documentation."
	@echo
	@echo " or just use sg('help') "
	@echo

doc-octave_modular:
	true

doc-python_modular:
	true

doc-r_modular:
	true

$(OBJFILES): ../.config .depend Makefile ../ChangeLog

$(ELWMS_TARGET): .depend $(OBJFILES) $(SRCFILES) $(HEADERFILES) $(OTHERDEPS)
	$(LINK) $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null) $(LINKFLAGS) -o $@ $(POSTLINKFLAGS)
	@$(POSTLINKCMD) $@
	@$(LIBSYMLINKCMD)

libshogun.a: .depend $(OBJFILES) $(SRCFILES) $(HEADERFILES)
	-ar rcs $@ $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null)

libshogunui.a: .depend $(OBJFILES) $(SRCFILES) $(HEADERFILES)
	-ar rcs $@ $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null)

%.R: %.${EXT_LIB_SWIG_ELWMS}
%.RData: %.R
	echo 'f="$<"; fdata="$@"; source(f); save(list=ls(all=TRUE),file=fdata, compress=TRUE); q(save="no")' | ($(LIBRARY_PATH)=../libshogun:../libshogunui R --no-save)

$(PRE_LIB_SWIG_ELWMS)%.${EXT_LIB_SWIG_ELWMS}:	%$(EXT_OBJ_SWIG)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(POSTLINKFLAGS)

%${EXT_OBJ_SWIG}:	%$(EXT_SRC_SWIG)
	$(COMP_CPP) $(COMPFLAGS_SWIG_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<

ifeq (yes,$(USE_DOXYGEN))
%${EXT_SRC_SWIG}:	%$(EXT_SWIG_DOXY) %.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $(DEFINES) $*.$(EXT_INP_SWIG)
	$(PYTHON) ../.scrub_docstrings.py $(EXT_IF_SWIG_ELWMS) $@

%$(EXT_SWIG_DOXY): %$(EXT_DOXY)
	$(DOXYGEN) $<
	$(PYTHON) ../.doxy2swig.py --quiet --no-function-definition $(shell basename $< $(EXT_DOXY))/doxygen_xml/index.xml $@
else
%${EXT_SRC_SWIG}: %.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $(DEFINES) $*.$(EXT_INP_SWIG)
	$(PYTHON) ../.scrub_docstrings.py $(EXT_IF_SWIG_ELWMS) $@
endif

# Do not include dependencies when they are about to be removed anyway.
ifneq ($(MAKECMDGOALS),distclean)
ifneq ($(MAKECMDGOALS),clean)

# Generate dependecies.
.depend:	../.config $(DEPDEPS)
ifeq (makedepend,$(MAKEDEPEND))
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_CPP):/g' >.depend
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_C):/g' >>.depend
else #gcc -MM
	find $(SRCDIR) -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)" \
	-exec $(MAKEDEPEND) $(DEFINES) $(INCLUDES) {} -MT {}.o \; >.depend
	( find . -name "*.$(EXT_INP_SWIG)" | cut -c 3- ; find ../shogun/ -name "*.h"  ) | \
			$(PYTHON) ../.generate_link_dependencies.py "$(PRE_LIB_SWIG_ELWMS)" "$(EXT_LIB_SWIG_ELWMS)" $(LIBSHOGUN_MODE) >>.depend
ifeq (r_modular,$(ELWMS_TARGET))
	    bash -c "( for i in \`grep '.so:' .depend | cut -f 1 -d ':'\` ; do echo \`echo \$$i | sed s/.so/.R:/\` \$$i ; done ) >>.depend"
endif

endif

# Include dependecies.
-include .depend
endif
endif
