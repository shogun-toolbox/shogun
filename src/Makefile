include Makefile.env

FILES := $(patsubst %.$(EXT_SRC),%.$(EXT_OBJ), $(wildcard */*.$(EXT_SRC)))
FILES2 := $(patsubst %.$(EXT_SRC2),%.$(EXT_OBJ2), $(wildcard */*.$(EXT_SRC2)))

.PHONY: all out clean doc texdoc htmldoc end condor purif profile depend

all:	out genefinder.`uname` end
out:
	@echo ============================================================================
	@echo ==== Building genefinder for platform $(UNAME) arch: $(ARCH)
	@echo ============================================================================ 
	@echo 
end:	
	@echo 
	@echo ============================================================================ 
	@echo ==== Done. Start with genefinder.$(UNAME)
	@echo ============================================================================ 
	@echo 
genefinder.`uname`: $(FILES) $(FILES2)
	$(LINK) -o genefinder.`uname` $(LINKFLAGS) */*.$(EXT_OBJ) */*.$(EXT_OBJ2)

debug: COMPFLAGS=$(COMPFLAGS_DEBUG) 
debug: COMPFLAGS2=$(COMPFLAGS_DEBUG)
debug: LINKFLAGS=$(LINKFLAGS_DEBUG) 
debug: COMPFLAGS2=$(COMPFLAGS_DEBUG)
debug: genefinder.`uname`
	gdb genefinder.`uname`

cplex: INCLUDES+=$(INCLUDES_CPLEX)
cplex: COMPFLAGS+=-DSVMCPLEX
cplex: COMPFLAGS2+=-DSVMCPLEX
cplex: LINKFLAGS+=$(LINKFLAGS_CPLEX)
cplex: genefinder.`uname`

static: LINKFLAGS+=-static
static: genefinder.`uname`

purif: COMP=$(COMP_SUN)
purif: LINK=$(LINK_SUN)
purif: COMPFLAGS=$(COMPFLAGS_SUN)
purif: LINKFLAGS=$(LINKFLAGS_SUN)
purif:	genefinder.`uname`
	/home/solaris2/purify/purify-4.2-solaris2/purify -always-use-cache-dir \
	-best-effort -cache-dir=/tmp/ccache $(COMP) */*.$(EXT_OBJ) \
	$(LINKFLAGS) -o genefinder.`uname`.purif

condor: COMP= $(COMP_CONDOR) 
condor: LINK= $(LINK_CONDOR) 
condor: COMPFLAGS= $(COMPFLAGS_CONDOR) 
condor: LINKFLAGS= $(LINKFLAGS_CONDOR)
condor: genefinder.`uname`

profile: COMPFLAGS= $(COMPFLAGS_PROF) 
profile: LINKFLAGS= $(LINKFLAGS_PROF)
profile: genefinder.`uname`
	 genefinder.`uname` && gprof genefinder.`uname` | less

doc:	texdoc htmldoc

texdoc:
	doc++ -l -ng -D 1 -X -B../doc/footer.dpp -o../doc/genefinder.tex -nd -p -v -t */*.h 
htmldoc:
	doc++ -B footer.dpp -nd -p -v --dir ../doc */*.h	
clean:
	rm -f */*.$(EXT_OBJ) */*.$(EXT_OBJ2) genefinder.* ../doc/*.html ../doc/genefinder.tex gmon.out 

depend: 
	makedepend $(INCLUDES) */*.h */*.$(EXT_SRC) */*.$(EXT_SRC2)

$(FILES): Makefile Makefile.env
$(FILES2): Makefile Makefile.env

%.$(EXT_OBJ):	%.$(EXT_SRC)
	$(COMP) $(COMPFLAGS) -c $(INCLUDES) -o $@ $<
%.$(EXT_OBJ2):	%.$(EXT_SRC2)
	$(COMP2) $(COMPFLAGS2) -c $(INCLUDES) -o $@ $<

# DO NOT DELETE


gui/GeneFinder.c.o: /usr/include/_G_config.h /usr/include/alloca.h
gui/GeneFinder.c.o: /usr/include/bits/confname.h /usr/include/bits/endian.h
gui/GeneFinder.c.o: /usr/include/bits/environments.h /usr/include/bits/fcntl.h
gui/GeneFinder.c.o: /usr/include/bits/huge_val.h /usr/include/bits/mathcalls.h
gui/GeneFinder.c.o: /usr/include/bits/mathdef.h /usr/include/bits/nan.h
gui/GeneFinder.c.o: /usr/include/bits/posix_opt.h
gui/GeneFinder.c.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/sched.h
gui/GeneFinder.c.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
gui/GeneFinder.c.o: /usr/include/bits/stat.h /usr/include/bits/stdio_lim.h
gui/GeneFinder.c.o: /usr/include/bits/time.h /usr/include/bits/types.h
gui/GeneFinder.c.o: /usr/include/bits/waitflags.h
gui/GeneFinder.c.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
gui/GeneFinder.c.o: /usr/include/bits/wordsize.h /usr/include/ctype.h
gui/GeneFinder.c.o: /usr/include/endian.h /usr/include/fcntl.h
gui/GeneFinder.c.o: /usr/include/features.h /usr/include/gconv.h
gui/GeneFinder.c.o: /usr/include/getopt.h /usr/include/gnu/stubs.h
gui/GeneFinder.c.o: /usr/include/libio.h /usr/include/math.h
gui/GeneFinder.c.o: /usr/include/stdio.h /usr/include/stdlib.h
gui/GeneFinder.c.o: /usr/include/string.h /usr/include/sys/cdefs.h
gui/GeneFinder.c.o: /usr/include/sys/select.h /usr/include/sys/stat.h
gui/GeneFinder.c.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
gui/GeneFinder.c.o: /usr/include/time.h /usr/include/unistd.h
gui/GeneFinder.c.o: /usr/include/wchar.h /usr/include/xlocale.h
gui/GeneFinder.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
gui/GeneFinder.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
gui/GeneFinder.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
gui/GeneFinder.c.o: hmm/HMM.h lib/Mathmatics.h lib/Observation.h lib/common.h
gui/GeneFinder.c.o: svm/Optimizer.h svm/SVM.h svm/SVM_light.h svm/kernel.h
hmm/HMM.c.o: /usr/include/_G_config.h /usr/include/alloca.h
hmm/HMM.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
hmm/HMM.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
hmm/HMM.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
hmm/HMM.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
hmm/HMM.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
hmm/HMM.c.o: /usr/include/bits/time.h /usr/include/bits/types.h
hmm/HMM.c.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
hmm/HMM.c.o: /usr/include/bits/wchar.h /usr/include/ctype.h
hmm/HMM.c.o: /usr/include/endian.h /usr/include/features.h
hmm/HMM.c.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
hmm/HMM.c.o: /usr/include/libio.h /usr/include/math.h /usr/include/stdio.h
hmm/HMM.c.o: /usr/include/stdlib.h /usr/include/sys/cdefs.h
hmm/HMM.c.o: /usr/include/sys/select.h /usr/include/sys/sysmacros.h
hmm/HMM.c.o: /usr/include/sys/types.h /usr/include/time.h /usr/include/wchar.h
hmm/HMM.c.o: /usr/include/xlocale.h
hmm/HMM.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
hmm/HMM.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
hmm/HMM.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
hmm/HMM.c.o: hmm/HMM.h lib/Mathmatics.h lib/Observation.h lib/common.h
hmm/HMM.o: /usr/include/_G_config.h /usr/include/alloca.h
hmm/HMM.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
hmm/HMM.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
hmm/HMM.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
hmm/HMM.o: /usr/include/bits/sched.h /usr/include/bits/select.h
hmm/HMM.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
hmm/HMM.o: /usr/include/bits/types.h /usr/include/bits/waitflags.h
hmm/HMM.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
hmm/HMM.o: /usr/include/endian.h /usr/include/features.h /usr/include/gconv.h
hmm/HMM.o: /usr/include/gnu/stubs.h /usr/include/libio.h /usr/include/math.h
hmm/HMM.o: /usr/include/stdio.h /usr/include/stdlib.h /usr/include/sys/cdefs.h
hmm/HMM.o: /usr/include/sys/select.h /usr/include/sys/sysmacros.h
hmm/HMM.o: /usr/include/sys/types.h /usr/include/time.h /usr/include/wchar.h
hmm/HMM.o: /usr/include/xlocale.h
hmm/HMM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
hmm/HMM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
hmm/HMM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
hmm/HMM.o: lib/Mathmatics.h lib/Observation.h lib/common.h
lib/Mathmatics.c.o: /usr/include/_G_config.h /usr/include/alloca.h
lib/Mathmatics.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
lib/Mathmatics.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
lib/Mathmatics.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
lib/Mathmatics.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
lib/Mathmatics.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
lib/Mathmatics.c.o: /usr/include/bits/time.h /usr/include/bits/types.h
lib/Mathmatics.c.o: /usr/include/bits/waitflags.h
lib/Mathmatics.c.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
lib/Mathmatics.c.o: /usr/include/endian.h /usr/include/features.h
lib/Mathmatics.c.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
lib/Mathmatics.c.o: /usr/include/libio.h /usr/include/math.h
lib/Mathmatics.c.o: /usr/include/stdio.h /usr/include/stdlib.h
lib/Mathmatics.c.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
lib/Mathmatics.c.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
lib/Mathmatics.c.o: /usr/include/time.h /usr/include/wchar.h
lib/Mathmatics.c.o: /usr/include/xlocale.h
lib/Mathmatics.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
lib/Mathmatics.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
lib/Mathmatics.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Mathmatics.c.o: lib/Mathmatics.h lib/common.h
lib/Mathmatics.o: /usr/include/alloca.h /usr/include/bits/endian.h
lib/Mathmatics.o: /usr/include/bits/huge_val.h /usr/include/bits/mathcalls.h
lib/Mathmatics.o: /usr/include/bits/mathdef.h /usr/include/bits/nan.h
lib/Mathmatics.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/sched.h
lib/Mathmatics.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
lib/Mathmatics.o: /usr/include/bits/types.h /usr/include/bits/waitflags.h
lib/Mathmatics.o: /usr/include/bits/waitstatus.h /usr/include/endian.h
lib/Mathmatics.o: /usr/include/features.h /usr/include/gnu/stubs.h
lib/Mathmatics.o: /usr/include/math.h /usr/include/stdlib.h
lib/Mathmatics.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
lib/Mathmatics.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
lib/Mathmatics.o: /usr/include/time.h /usr/include/xlocale.h
lib/Mathmatics.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
lib/Mathmatics.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Mathmatics.o: lib/common.h
lib/Observation.c.o: /usr/include/_G_config.h /usr/include/alloca.h
lib/Observation.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
lib/Observation.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
lib/Observation.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
lib/Observation.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
lib/Observation.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
lib/Observation.c.o: /usr/include/bits/types.h /usr/include/bits/waitflags.h
lib/Observation.c.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
lib/Observation.c.o: /usr/include/endian.h /usr/include/features.h
lib/Observation.c.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
lib/Observation.c.o: /usr/include/libio.h /usr/include/math.h
lib/Observation.c.o: /usr/include/stdio.h /usr/include/stdlib.h
lib/Observation.c.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
lib/Observation.c.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
lib/Observation.c.o: /usr/include/time.h /usr/include/wchar.h
lib/Observation.c.o: /usr/include/xlocale.h
lib/Observation.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
lib/Observation.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
lib/Observation.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Observation.c.o: lib/Mathmatics.h lib/Observation.h lib/common.h
lib/Observation.o: /usr/include/_G_config.h /usr/include/alloca.h
lib/Observation.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
lib/Observation.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
lib/Observation.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
lib/Observation.o: /usr/include/bits/sched.h /usr/include/bits/select.h
lib/Observation.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
lib/Observation.o: /usr/include/bits/types.h /usr/include/bits/waitflags.h
lib/Observation.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
lib/Observation.o: /usr/include/endian.h /usr/include/features.h
lib/Observation.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
lib/Observation.o: /usr/include/libio.h /usr/include/math.h
lib/Observation.o: /usr/include/stdio.h /usr/include/stdlib.h
lib/Observation.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
lib/Observation.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
lib/Observation.o: /usr/include/time.h /usr/include/wchar.h
lib/Observation.o: /usr/include/xlocale.h
lib/Observation.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
lib/Observation.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
lib/Observation.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Observation.o: lib/Mathmatics.h lib/common.h
svm/Optimizer.c.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/Optimizer.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/Optimizer.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/Optimizer.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/Optimizer.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/Optimizer.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/Optimizer.c.o: /usr/include/bits/time.h /usr/include/bits/types.h
svm/Optimizer.c.o: /usr/include/bits/waitflags.h
svm/Optimizer.c.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
svm/Optimizer.c.o: /usr/include/ctype.h /usr/include/endian.h
svm/Optimizer.c.o: /usr/include/features.h /usr/include/gconv.h
svm/Optimizer.c.o: /usr/include/gnu/stubs.h /usr/include/libio.h
svm/Optimizer.c.o: /usr/include/math.h /usr/include/stdio.h
svm/Optimizer.c.o: /usr/include/stdlib.h /usr/include/string.h
svm/Optimizer.c.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
svm/Optimizer.c.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
svm/Optimizer.c.o: /usr/include/time.h /usr/include/wchar.h
svm/Optimizer.c.o: /usr/include/xlocale.h
svm/Optimizer.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/Optimizer.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/Optimizer.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/Optimizer.c.o: lib/Mathmatics.h lib/Observation.h lib/common.h
svm/Optimizer.c.o: svm/Optimizer.h svm/SVM.h svm/pr_loqo.h
svm/Optimizer.o: lib/common.h
svm/SVM.c.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/SVM.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/SVM.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/SVM.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/SVM.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/SVM.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/SVM.c.o: /usr/include/bits/time.h /usr/include/bits/types.h
svm/SVM.c.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
svm/SVM.c.o: /usr/include/bits/wchar.h /usr/include/ctype.h
svm/SVM.c.o: /usr/include/endian.h /usr/include/features.h
svm/SVM.c.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
svm/SVM.c.o: /usr/include/libio.h /usr/include/math.h /usr/include/stdio.h
svm/SVM.c.o: /usr/include/stdlib.h /usr/include/string.h
svm/SVM.c.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
svm/SVM.c.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
svm/SVM.c.o: /usr/include/time.h /usr/include/wchar.h /usr/include/xlocale.h
svm/SVM.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM.c.o: lib/Mathmatics.h lib/Observation.h lib/common.h svm/SVM.h
svm/SVM.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/SVM.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/SVM.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/SVM.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/SVM.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/SVM.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/SVM.o: /usr/include/bits/time.h /usr/include/bits/types.h
svm/SVM.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
svm/SVM.o: /usr/include/bits/wchar.h /usr/include/ctype.h
svm/SVM.o: /usr/include/endian.h /usr/include/features.h /usr/include/gconv.h
svm/SVM.o: /usr/include/gnu/stubs.h /usr/include/libio.h /usr/include/math.h
svm/SVM.o: /usr/include/stdio.h /usr/include/stdlib.h /usr/include/string.h
svm/SVM.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
svm/SVM.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
svm/SVM.o: /usr/include/time.h /usr/include/wchar.h /usr/include/xlocale.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM.o: lib/Mathmatics.h lib/Observation.h lib/common.h
svm/SVM_light.c.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/SVM_light.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/SVM_light.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/SVM_light.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/SVM_light.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/SVM_light.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/SVM_light.c.o: /usr/include/bits/time.h /usr/include/bits/types.h
svm/SVM_light.c.o: /usr/include/bits/waitflags.h
svm/SVM_light.c.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
svm/SVM_light.c.o: /usr/include/ctype.h /usr/include/endian.h
svm/SVM_light.c.o: /usr/include/features.h /usr/include/gconv.h
svm/SVM_light.c.o: /usr/include/gnu/stubs.h /usr/include/libio.h
svm/SVM_light.c.o: /usr/include/math.h /usr/include/stdio.h
svm/SVM_light.c.o: /usr/include/stdlib.h /usr/include/string.h
svm/SVM_light.c.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
svm/SVM_light.c.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
svm/SVM_light.c.o: /usr/include/time.h /usr/include/wchar.h
svm/SVM_light.c.o: /usr/include/xlocale.h
svm/SVM_light.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM_light.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM_light.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM_light.c.o: hmm/HMM.h lib/Mathmatics.h lib/Observation.h lib/common.h
svm/SVM_light.c.o: svm/Optimizer.h svm/SVM.h svm/SVM_light.h svm/kernel.h
svm/SVM_light.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/SVM_light.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/SVM_light.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/SVM_light.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/SVM_light.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/SVM_light.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/SVM_light.o: /usr/include/bits/time.h /usr/include/bits/types.h
svm/SVM_light.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
svm/SVM_light.o: /usr/include/bits/wchar.h /usr/include/ctype.h
svm/SVM_light.o: /usr/include/endian.h /usr/include/features.h
svm/SVM_light.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
svm/SVM_light.o: /usr/include/libio.h /usr/include/math.h /usr/include/stdio.h
svm/SVM_light.o: /usr/include/stdlib.h /usr/include/string.h
svm/SVM_light.o: /usr/include/sys/cdefs.h /usr/include/sys/select.h
svm/SVM_light.o: /usr/include/sys/sysmacros.h /usr/include/sys/types.h
svm/SVM_light.o: /usr/include/time.h /usr/include/wchar.h
svm/SVM_light.o: /usr/include/xlocale.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM_light.o: hmm/HMM.h lib/Mathmatics.h lib/Observation.h lib/common.h
svm/SVM_light.o: svm/Optimizer.h svm/SVM.h svm/kernel.h
svm/kernel.c.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/kernel.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/kernel.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/kernel.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/kernel.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/kernel.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/kernel.c.o: /usr/include/bits/types.h /usr/include/bits/waitflags.h
svm/kernel.c.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
svm/kernel.c.o: /usr/include/endian.h /usr/include/features.h
svm/kernel.c.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
svm/kernel.c.o: /usr/include/libio.h /usr/include/math.h /usr/include/stdio.h
svm/kernel.c.o: /usr/include/stdlib.h /usr/include/sys/cdefs.h
svm/kernel.c.o: /usr/include/sys/select.h /usr/include/sys/sysmacros.h
svm/kernel.c.o: /usr/include/sys/types.h /usr/include/time.h
svm/kernel.c.o: /usr/include/wchar.h /usr/include/xlocale.h
svm/kernel.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/kernel.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/kernel.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/kernel.c.o: hmm/HMM.h lib/Mathmatics.h lib/Observation.h lib/common.h
svm/kernel.c.o: svm/kernel.h
svm/kernel.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/kernel.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/kernel.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/kernel.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/kernel.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/kernel.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/kernel.o: /usr/include/bits/types.h /usr/include/bits/waitflags.h
svm/kernel.o: /usr/include/bits/waitstatus.h /usr/include/bits/wchar.h
svm/kernel.o: /usr/include/endian.h /usr/include/features.h
svm/kernel.o: /usr/include/gconv.h /usr/include/gnu/stubs.h
svm/kernel.o: /usr/include/libio.h /usr/include/math.h /usr/include/stdio.h
svm/kernel.o: /usr/include/stdlib.h /usr/include/sys/cdefs.h
svm/kernel.o: /usr/include/sys/select.h /usr/include/sys/sysmacros.h
svm/kernel.o: /usr/include/sys/types.h /usr/include/time.h
svm/kernel.o: /usr/include/wchar.h /usr/include/xlocale.h
svm/kernel.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/kernel.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/kernel.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/kernel.o: hmm/HMM.h lib/Mathmatics.h lib/Observation.h lib/common.h
svm/pr_loqo.c.o: /usr/include/_G_config.h /usr/include/alloca.h
svm/pr_loqo.c.o: /usr/include/bits/endian.h /usr/include/bits/huge_val.h
svm/pr_loqo.c.o: /usr/include/bits/mathcalls.h /usr/include/bits/mathdef.h
svm/pr_loqo.c.o: /usr/include/bits/nan.h /usr/include/bits/pthreadtypes.h
svm/pr_loqo.c.o: /usr/include/bits/sched.h /usr/include/bits/select.h
svm/pr_loqo.c.o: /usr/include/bits/sigset.h /usr/include/bits/stdio_lim.h
svm/pr_loqo.c.o: /usr/include/bits/time.h /usr/include/bits/types.h
svm/pr_loqo.c.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
svm/pr_loqo.c.o: /usr/include/bits/wchar.h /usr/include/endian.h
svm/pr_loqo.c.o: /usr/include/features.h /usr/include/gconv.h
svm/pr_loqo.c.o: /usr/include/gnu/stubs.h /usr/include/libio.h
svm/pr_loqo.c.o: /usr/include/math.h /usr/include/stdio.h
svm/pr_loqo.c.o: /usr/include/stdlib.h /usr/include/sys/cdefs.h
svm/pr_loqo.c.o: /usr/include/sys/select.h /usr/include/sys/sysmacros.h
svm/pr_loqo.c.o: /usr/include/sys/types.h /usr/include/time.h
svm/pr_loqo.c.o: /usr/include/wchar.h /usr/include/xlocale.h
svm/pr_loqo.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/pr_loqo.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/pr_loqo.c.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/pr_loqo.c.o: lib/common.h svm/pr_loqo.h
