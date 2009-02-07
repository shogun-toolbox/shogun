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
	@+$(foreach part, $(PARTS), make -C $(part) $@ &&) true
	@echo 
	@echo ============================================================
	@echo ==== Done. Check the subdirectories for binaries.
	@echo ============================================================
	@echo 
endif

install:  $(PARTDEPS)
	+$(foreach part, $(PARTS), make -C $(part) install-$(part) &&) true

tests: run-testsuite check-examples

run-testsuite: $(PARTDEPTS)
	+$(foreach part, $(PARTS), make -C $(part) run-testsuite-$(part) &&) true

check-examples: $(PARTDEPTS)
	+$(foreach part, $(PARTS), make -C $(part) check-examples-$(part) &&) true

doc:  $(PARTDEPS)
	+$(foreach part, $(PARTS), make -C $(part) $@ &&) true

distclean: clean
	@+$(foreach part, $(PARTS), make -C $(part) $@ &&) true
	@rm -f .config tags ui shogun libshogunui/shogun
	@rm -f */Makefile

clean:  $(PARTDEPS)
	@+$(foreach part, $(PARTS), make -C $(part) $@ &&) true
	@rm -f configure.log .cpuinfo ./configure-* cplex.log

dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

$(PARTDEPS): Makefile.template
	@+$(foreach part, $(PARTS), cat Makefile.template | sed 's/TEMPLATE/$(shell echo $(part) | tr 'a-z' 'A-Z')/g' >$(part)/Makefile; )


ChangeLog:
	@if [ -n "`which svn2cl`" ]; then \
		svn2cl --group-by-day --authors=.authors 2>/dev/null; \
	else \
		echo "Need svn2cl (from subversion-tools) to update ChangeLog!"; \
		exit 1; \
	fi
