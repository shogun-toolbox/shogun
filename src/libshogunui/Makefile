.SUFFIXES: .c .c.o .cpp .cpp.o .h .so 

ifeq (GNU,$(shell make --version 2>&1 | grep GNU >/dev/null && echo GNU))
MAKE=make
else
MAKE=gmake
endif

all:	just-do-it

include ../.config
ifeq (../.config-local,$(wildcard ../.config-local))
include ../.config-local
endif

INCLUDES += $(INCLUDES_LIBSHOGUNUI) -I../libshogun
DEFINES += $(DEFINES_LIBSHOGUNUI)
LINKFLAGS += $(LINKFLAGS_LIBSHOGUNUI)
POSTLINKFLAGS += $(POSTLINKFLAGS_LIBSHOGUNUI)

SRCDIR = .

# SRC/OBJFILES lie all in the subdir of SRCDIR
HEADERFILES	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_HEADER)" )
SRCFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)")
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)")) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)"))

.PHONY: all out clean end targets warn2err err2warn ctags depgen buildout buildend dist

ifeq (.depend,$(wildcard .depend))
include .depend
just-do-it:	buildout $(LIBSHOGUNUI_TARGET) buildend
else
just-do-it:	depgen .depend
		test -f .depend && $(MAKE)
endif

depgen:	../.config
	@echo =================================
	@echo ==== Generating dependencies ====
	@echo =================================

buildout:
	@echo ========================================================================================
	@echo ==== Building shogun for platform $(KERNELNAME)-$(MACHINE) target $(LIBSHOGUNUI_TARGET)
	@echo ==== opts: $(COMPFLAGS_CPP)
	@echo ========================================================================================
	@echo 
buildend:	$(LIBSHOGUNUI_TARGET)	
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
		@rm -f tags 

clean:
	@rm -f $(shell find . -name "*.cpp.o" -o -name "*.c.o" -o -name "*.so" -o -name "*.dll" -o -name "*~" -o -name ".#*" 2>/dev/null)
	@rm -f $(shell find . -type f -name '*.swp')
	@rm -f libshogunui.a libshogunui.so*

dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

install: $(LIBSHOGUNUI_TARGET)
	install -d -m755 $(DESTDIR)$(LIBDIR)
	install -d -m755 "$(DESTDIR)$(INCDIR)/shogun/ui"
	install -m755 $(LIBSHOGUNUI_TARGET) $(DESTDIR)$(LIBDIR)
	install -m755 libshogunui.so $(DESTDIR)$(LIBDIR)
	install -m755 libshogunui.a $(DESTDIR)$(LIBDIR)
	install -m644 *.h -t "$(DESTDIR)$(INCDIR)/shogun/ui"

doc:
		-@sed -i "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = `sed -n 's/^#define VERSION_RELEASE "\(.*\)_"/\1/p' $@`/" ../doc/Doxyfile
		cd ../doc && make

$(OBJFILES): ../.config .depend ../ChangeLog

dep:	depend
depend:		../.config
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
endif

.depend:	../.config
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
endif

$(LIBSHOGUNUI_TARGET): $(OBJFILES) $(SRCFILES) $(HEADERFILES)
	$(LINK) $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null) $(LINKFLAGS) -o $(LIBSHOGUNUI_TARGET) $(POSTLINKFLAGS)
	-ar rs libshogunui.a $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null)
	rm -f libshogunui.so
	ln -s $(LIBSHOGUNUI_TARGET) libshogunui.so

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<
