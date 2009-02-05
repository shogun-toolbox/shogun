# include config options
#
#
.SUFFIXES: .c .c.o .cpp .cpp.o .h .i .so .oct _wrap.cxx _wrap.cxx.o

ifeq (GNU,$(shell make --version 2>&1 | grep GNU >/dev/null && echo GNU))
MAKE=make
else
MAKE=gmake
endif

all:	just-do-it

ifeq (.config,$(wildcard .config))
include .config

ifeq (.config-local,$(wildcard .config-local))
include .config-local
endif

just-do-it: 
	make -C libshogun $@
	make -C libshogunui $@
	make -C interfaces $@
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
	@echo For additional options see
	@echo "              ./configure --help"
	@echo
	@exit 1
#runconfigurehunkend
endif

install: 
	make -C libshogun $@
	make -C libshogunui $@
	make -C interfaces $@
doc: 
	make -C libshogun $@
	make -C libshogunui $@
	make -C interfaces $@

SRCDIR = .

distclean: clean
		@rm -f .config tags lib/config.h libshogun/lib/versionstring.h
swigclean:
	@rm -f $(shell find -name "*_wrap.h" -o -name "*_wrap.cxx.o" -o -name "*_wrap.cxx" 2>/dev/null) 

swigdocclean:
	@rm -rf $(shell find -name doxygen_xml -o -name "*_doxygen.i" 2>/dev/null) 

clean: swigclean swigdocclean
	@rm -f $(shell find . -name "*.cpp.o" -o -name "*.c.o" -o -name "*.so" -o -name "*.dll" -o -name "*~" -o -name ".#*" 2>/dev/null)
	@rm -f $(shell find . -type f -name '*.py*' ! -name '__*.py' ! -name '.*.py' | grep -v svn)
	@rm -f $(shell find . -type f -name '*.R')
	@rm -f $(shell find . -type f -name '*.RData')
	@rm -f $(shell find . -type f -name '*.oct')
	@rm -f $(shell find . -type f -name '*.swp')
	@rm -f $(shell find . -type f -name 'octave-core')
	@rm -f sg.mex* shogun shogun.exe gmon.out .depend configure.log .cpuinfo ./configure-* cplex.log
	@rm -f libshogun.a

dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

tests: run-testsuite check-examples

# python-modular
ifeq (python-modular,$(INTERFACE))
run-testsuite:
		cd ../testsuite/$(INTERFACE) && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./test_all.sh	)
check-examples:
		cd ../examples/$(INTERFACE) && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./check.sh	)
else
ifeq (python,$(INTERFACE))
run-testsuite:
		cd ../testsuite/$(INTERFACE) && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./test_all.sh	)
check-examples:
		cd ../examples/$(INTERFACE) && ( PYTHON=$(PYTHON) PYTHONPATH="$(DESTDIR)$(PYDIR)" ./check.sh	)
else
# R-modular
ifeq (r-modular,$(INTERFACE))
run-testsuite:
		cd ../testsuite/$(INTERFACE) && ( R_LIBS="$(DESTDIR)$(RDIR)" ./test_all.sh )
check-examples:
		cd ../examples/$(INTERFACE) && ( R_LIBS="$(DESTDIR)$(RDIR)" ./check.sh )
else
# R
ifeq (r,$(INTERFACE))
run-testsuite:
		cd ../testsuite/$(INTERFACE) && ( R_LIBS="$(DESTDIR)$(RDIR)" ./test_all.sh )
check-examples:
		cd ../examples/$(INTERFACE) && ( R_LIBS="$(DESTDIR)$(RDIR)" ./check.sh )
else
# Standalone
ifeq (cmdline,$(INTERFACE))
run-testsuite:
		cd ../testsuite/$(INTERFACE) && ( PATH="$(DESTDIR)$(BINDIR)"  ./test_all.sh )
check-examples:
		cd ../examples/$(INTERFACE) && ( PATH="$(DESTDIR)$(BINDIR)"  ./check.sh )
# Octave
else
ifeq (octave-modular,$(INTERFACE))
run-testsuite:
	cd ../testsuite/$(INTERFACE) && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)/shogun:" ./test_all.sh )
check-examples:
	cd ../examples/$(INTERFACE) && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)/shogun:" ./check.sh )
else
ifeq (octave,$(INTERFACE))
run-testsuite:
	cd ../testsuite/$(INTERFACE) && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR):" ./test_all.sh - octave )
check-examples:
	cd ../examples/$(INTERFACE) && ( OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR):" ./check.sh )
#Matlab
else
ifeq (matlab,$(INTERFACE))
run-testsuite:
	cd ../testsuite/$(INTERFACE) && ( ln -sf ../../src/$(TARGET) ; ./test_all.sh - matlab)
check-examples:
	cd ../examples/$(INTERFACE) && ./check.sh
endif
endif
endif
endif
endif
endif
endif
endif

ChangeLog:
	@if [ -n "`which svn2cl`" ]; then \
		svn2cl --group-by-day --authors=.authors 2>/dev/null; \
	else \
		echo "Need svn2cl (from subversion-tools) to update ChangeLog!"; \
		exit 1; \
	fi
