ifneq (.config,$(wildcard .config))
all:
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
else
include .config
ifeq (.config-local,$(wildcard .config-local))
include .config-local
endif
PARTDEPS = $(foreach part, $(PARTS), $(part)/Makefile )
all:  $(PARTDEPS)
	@echo ============================================================
	@echo ==== Building shogun for platform $(KERNELNAME)-$(MACHINE)
	@echo ==== opts: $(COMPFLAGS_CPP)
	@echo ============================================================
	@echo 
	+$(foreach part, $(PARTS), make -C $(part) $@ &&) true
	@echo 
	@echo ============================================================
	@echo ==== Done. Check the subdirectories for binaries
	@echo ============================================================
	@echo 
endif

install:  $(PARTDEPS)
	+$(foreach part, $(PARTS), make -C $(part) $@ &&) true

doc:  $(PARTDEPS)
	+$(foreach part, $(PARTS), make -C $(part) $@ &&) true

distclean: clean
	+$(foreach part, $(PARTS), make -C $(part) $@ &&) true
	@rm -f .config tags ui shogun libshogunui/shogun
	@rm -f */Makefile

clean:  $(PARTDEPS)
	+$(foreach part, $(PARTS), make -C $(part) $@ &&) true
	@rm -f configure.log .cpuinfo ./configure-* cplex.log

dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

$(PARTDEPS): Makefile.template
	+$(foreach part, $(PARTS), cat Makefile.template | sed 's/TEMPLATE/$(shell echo $(part) | tr 'a-z' 'A-Z')/g' >$(part)/Makefile; )

tests: run-testsuite check-examples

# python-modular
ifeq (yes,$(INTERFACE))
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
