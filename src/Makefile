.PHONY: ChangeLog unit-tests

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
PARTDEPS = $(foreach part, $(PARTS), interfaces/$(part)/Makefile )
all:  $(PARTDEPS) examples $(LIBSHOGUN_TARGET)
	@echo ============================================================
	@echo ==== Building Shogun for platform $(KERNELNAME)-$(MACHINE)
	@echo ==== Compilation options: 
	@echo $(COMPFLAGS_CPP) | fold -s -w 60 | sed 's/.*/==== &/'
	@echo ============================================================
	@echo 
	$(foreach part, $(PARTS), $(MAKE) -C interfaces/$(part) $@ &&) true
	@echo 
	@echo ============================================================
	@echo ==== Done. Check the subdirectories for binaries.
	@echo ============================================================
	@echo 
endif

partdeps: shogun/Makefile $(PARTDEPS)

ctags: shogun/Makefile
	$(MAKE) -C shogun ctags
	ln -s shogun/tags

$(LIBSHOGUN_TARGET): shogun/Makefile 
	$(MAKE) -C shogun

shogun/Makefile: Makefile.template
	@cat Makefile.template | sed 's/template/libshogun/g' | \
		sed 's/TEMPLATE/LIBSHOGUN/g' >$@;

examples:
	$(MAKE) -C../examples
	touch $@

install:  $(PARTDEPS) examples
	$(MAKE) -C shogun install
	+$(foreach part, $(PARTS), $(MAKE) -C interfaces/$(part) install-$(part) &&) true

uninstall:
	@test -z "$(LIBDIR)" || rm -f "$(DESTDIR)$(LIBDIR)/libshogun"*
	@test -z "$(INCDIR)" || rm -rf "$(DESTDIR)$(INCDIR)/shogun"
	@test -z "$(BINDIR)" || rm -rf "$(DESTDIR)$(BINDIR)/shogun"
	@test -z "$(PYDIR)" || rm -rf "$(DESTDIR)$(PYDIR)/sg.so"
	@test -z "$(PYDIR)" || rm -rf "$(DESTDIR)$(PYDIR)/sg.dylib"
	@test -z "$(PYDIR)" || rm -rf "$(DESTDIR)$(PYDIR)/shogun"
	@test -z "$(RDIR)" || rm -rf "$(DESTDIR)$(RDIR)/shogun"
	@test -z "$(OCTDIR)" || rm -rf "$(DESTDIR)$(OCTDIR)/shogun"

unit-tests: .config
ifneq ($(USE_GMOCK),)
	$(MAKE) -C ../tests/unit
endif

tests:
	$(MAKE) check-examples
	$(MAKE) run-testsuite

run-testsuite: $(PARTDEPS)
	+$(foreach part, $(PARTS), $(MAKE) -C interfaces/$(part) run-testsuite-$(part) &&) true

check-examples: $(PARTDEPS)
	$(MAKE) -C shogun check-examples
	+$(foreach part, $(PARTS), $(MAKE) -C interfaces/$(part) check-examples-$(part) &&) true

install-local:
	$(MAKE) DESTDIR="$(shell readlink -f .)"/build-local install

clean-local:
	@rm -rf build-local

%-local: install-local
	$(MAKE) DESTDIR="$(shell readlink -f .)"/build-local $*

doc:  $(PARTDEPS)
	+$(foreach part, $(PARTS), $(MAKE) -C interfaces/$(part) doc-$(part) &&) true
tutorial:
	$(MAKE) -C ../doc/tutorial

distclean: clean
	$(MAKE) -C shogun distclean
	@+$(foreach part, $(PARTS), $(MAKE) -C interfaces/$(part) $@ &&) true
	@rm -f .config tags
	@rm -f */Makefile interfaces/*/Makefile

clean:  $(PARTDEPS)
	$(MAKE) -C shogun clean
	@+$(foreach part, $(PARTS), $(MAKE) -C interfaces/$(part) $@ &&) true
	@rm -rf configure-*.dSYM
	@rm -f configure.log .cpuinfo .cpuinfo.c ./configure-* cplex.log examples
	$(MAKE) clean-local

dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

$(PARTDEPS): Makefile.template
	@+$(foreach part, $(PARTS), cat Makefile.template | sed 's/template/$(part)/g' | \
		sed 's/TEMPLATE/$(shell echo $(part) | tr 'a-z' 'A-Z')/g' >interfaces/$(part)/Makefile; )

linecount:
	@echo -n "C/C++ source code files (.cpp,.c):"
	@cat `find ./ -type f -name '*.cpp' -o -name '*.c'`  | grep -vc '^ *$$'
	@echo -n "C/C++ header files (.h):"
	@cat `find ./ -type f -name '*.h'`  | grep -vc '^ *$$'
	@echo -n "SWIG header files (.i):"
	@cat `find ./interfaces -type f -name '*.i'`  | grep -vc '^ *$$'
	@echo -n "Total:"
	@(cat `find ./ -type f -name '*.cpp' -o -name '*.c' -o -name '*.h'`; cat `find ./interfaces -type f -name '*.i'`)  | grep -vc '^ *$$'

ChangeLog:
	git log --name-status --graph >$@
