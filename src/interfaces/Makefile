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

include ../.config

ifeq (../.config-local,$(wildcard ../.config-local))
include ../.config-local
endif

INCLUDES += -I.. -I../libshogun
COMPFLAGS_CPP += $(DEFINES_CMDLINE)
#DEFINES += $(DEFINES_LIBSHOGUN)
LINKFLAGS += -L../libshogun -lshogun -L../libshogunui -lshogunui -lreadline
#POSTLINKFLAGS += $(POSTLINKFLAGS_LIBSHOGUN)

SRCDIR = .

# SRC/OBJFILES lie all in the subdir of SRCDIR
HEADERFILES	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_HEADER)" )
SRCFILES 	= $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)")
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_CPP)")) $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)"))

.PHONY: config just-do-it all clean dep depend

ifeq (.depend,$(wildcard .depend))
include .depend
just-do-it:	$(CMDLINE_TARGET)
else
just-do-it:	.depend
		test -f .depend && $(MAKE)
endif

distclean: clean
		@rm -f tags lib/config.h libshogun/lib/versionstring.h

clean:
	@rm -f $(shell find . -name "*.cpp.o" -o -name "*.c.o" -o -name "*.so" -o -name "*.dll" -o -name "*~" -o -name ".#*" 2>/dev/null)
	@rm -f $(shell find . -type f -name '*.py*' ! -name '__*.py' ! -name '.*.py' | grep -v svn)
	@rm -f $(shell find . -type f -name '*.R')
	@rm -f $(shell find . -type f -name '*.RData')
	@rm -f $(shell find . -type f -name '*.oct')
	@rm -f $(shell find . -type f -name '*.swp')
	@rm -f $(shell find . -type f -name 'octave-core')
	@rm -f sg.mex* shogun shogun.exe gmon.out .depend configure.log .cpuinfo ./configure-* cplex.log
	@rm -f libshogun.a

dep:	depend
depend:		../.config ../libshogun/lib/versionstring.h
ifeq (makedepend,$(MAKEDEPEND))
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_CPP)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_CPP):/g' >.depend
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_C):/g' >>.depend
else #gcc -MM
	find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)" \
	-exec $(MAKEDEPEND) $(DEFINES) $(INCLUDES) {} -MT {}.o \; >.depend
endif

.depend:	../.config ../libshogun/lib/versionstring.h
ifeq (makedepend,$(MAKEDEPEND))
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_CPP)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_CPP):/g' >.depend
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_C):/g' >>.depend
else #gcc -MM
	find $(SRCDIR) -mindepth 2 -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)" \
	-exec $(MAKEDEPEND) $(DEFINES) $(INCLUDES) {} -MT {}.o \; >.depend
endif

$(OBJFILES): ../.config .depend ../libshogun/lib/versionstring.h ../ChangeLog

$(CMDLINE_TARGET): $(OBJFILES) $(SRCFILES) $(HEADERFILES)
	$(LINK) $(shell find $(SRCDIR) -mindepth 2 -name "*.$(EXT_OBJ_CPP)" -o -name "*.$(EXT_OBJ_C)" 2>/dev/null) $(LINKFLAGS) -o $@ $(POSTLINKFLAGS)
	$(POSTLINKCMD) $@

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(COMPFLAGS_CPP) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(COMPFLAGS_C) -c $(INCLUDES) -o $@ $<
