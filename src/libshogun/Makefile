.SUFFIXES: .c .c.o .cpp .cpp.o .h .so 

ifeq (GNU,$(shell make --version 2>&1 | grep GNU >/dev/null && echo GNU))
MAKE=make
else
MAKE=gmake
endif

include ../.config
ifeq (../.config-local,$(wildcard ../.config-local))
include ../.config-local
endif

SRCDIR = .

# SRC/OBJFILES lie all in the subdir of SRCDIR
HEADERFILES	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_HEADER)" )
SRCFILES 	= $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)")
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_CPP)")) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)"))

.PHONY: all out clean end targets warn2err err2warn ctags depgen buildout buildend dist

just-do-it:	buildout $(LIBSHOGUN_TARGET) buildend

buildout:
	@echo ========================================================================================
	@echo ==== Building shogun for platform $(KERNELNAME)-$(MACHINE) target $(LIBSHOGUN_TARGET)
	@echo ==== opts: $(COMPFLAGS_CPP)
	@echo ========================================================================================
	@echo 
buildend:	$(LIBSHOGUN_TARGET)	
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
	@rm -f libshogun.a libshogun.so*

dist:	distclean
	rm -f $(TAR).gz
	tar -cvf $(TAR) -C ../../ shogun/src && gzip -9 $(TAR)

install: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(LIBDIR)
	install -d -m755 "$(DESTDIR)$(INCDIR)/shogun"
	install -m755 libshogun.so $(DESTDIR)$(LIBDIR)
	install -m755 libshogun.a $(DESTDIR)$(LIBDIR)

	find ./ -mindepth 1 -type d ! \( -path '*.svn*' -o -name '*interface*' -o -name '*guilib*' -o -name '*doxygen*' \)\
		-exec install -d -m755 "$(DESTDIR)$(INCDIR)/shogun/"\{\} \;
	find ./ -mindepth 1 -type d ! \( -path '*.svn*' -o -name '*wrap*' -o -name '*interface*' -o -name '*guilib*' -o -name '*doxygen*' \) \
		-exec sh -c "install -m644 {}/*.h -t $(DESTDIR)$(INCDIR)/shogun/{}" \;

doc:	lib/versionstring.h
		-@sed -i "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = `sed -n 's/^#define VERSION_RELEASE "\(.*\)_"/\1/p' $@`/" ../doc/Doxyfile
		cd ../doc && make

$(OBJFILES): ../.config ../.depend lib/versionstring.h ../ChangeLog

$(LIBSHOGUN_TARGET): $(OBJFILES) $(SRCFILES) $(HEADERFILES)
	$(LINK) $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null) $(LINKFLAGS) -o libshogun.so $(POSTLINKFLAGS)
	-ar rs libshogun.a $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null)

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<
