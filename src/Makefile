include Makefile.env

FILES := $(patsubst %.$(EXT_SRC),%.$(EXT_OBJ), $(wildcard */*.$(EXT_SRC)))
FILES2 := $(patsubst %.$(EXT_SRC2),%.$(EXT_OBJ2), $(wildcard */*.$(EXT_SRC2)))

.PHONY: all out clean doc texdoc htmldoc end condor purif profile depend targets

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

targets:
	@echo "=========="
	@echo "=targets:="
	@echo "=========="
	@egrep '^[a-z]+:' Makefile | grep -v '\.o:' | sed -e 's/:.*//g' -e 's/out//g' -e 's/end//g' -e 's/^/     /g' | sort | uniq
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
	makedepend $(INCLUDES) */*.h */*.$(EXT_SRC) -o.$(EXT_OBJ)
	makedepend -a $(INCLUDES) */*.h */*.$(EXT_SRC2) -o.$(EXT_OBJ2)

$(FILES): Makefile Makefile.env
$(FILES2): Makefile Makefile.env

%.$(EXT_OBJ):	%.$(EXT_SRC)
	$(COMP) $(COMPFLAGS) -c $(INCLUDES) -o $@ $<
%.$(EXT_OBJ2):	%.$(EXT_SRC2)
	$(COMP2) $(COMPFLAGS2) -c $(INCLUDES) -o $@ $<

# DO NOT DELETE

hmm/HMM.cpp.o: lib/Mathmatics.h lib/common.h /usr/include/math.h
hmm/HMM.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
hmm/HMM.cpp.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
hmm/HMM.cpp.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
hmm/HMM.cpp.o: /usr/include/stdio.h
hmm/HMM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
hmm/HMM.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
hmm/HMM.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
hmm/HMM.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
hmm/HMM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
hmm/HMM.cpp.o: /usr/include/bits/stdio_lim.h lib/Observation.h lib/io.h
lib/io.cpp.o: /usr/include/stdio.h /usr/include/features.h
lib/io.cpp.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/io.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/io.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
lib/io.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/io.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/io.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/io.cpp.o: /usr/include/bits/stdio_lim.h
lib/Mathmatics.cpp.o: lib/common.h /usr/include/math.h
lib/Mathmatics.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
lib/Mathmatics.cpp.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
lib/Mathmatics.cpp.o: /usr/include/bits/mathdef.h
lib/Mathmatics.cpp.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
lib/Mathmatics.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/Mathmatics.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Mathmatics.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Mathmatics.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Mathmatics.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/Mathmatics.cpp.o: /usr/include/bits/stdio_lim.h
lib/Observation.cpp.o: /usr/include/stdio.h /usr/include/features.h
lib/Observation.cpp.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/Observation.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/Observation.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Observation.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Observation.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Observation.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/Observation.cpp.o: /usr/include/bits/stdio_lim.h lib/io.h lib/common.h
lib/Observation.cpp.o: lib/Mathmatics.h /usr/include/math.h
lib/Observation.cpp.o: /usr/include/bits/huge_val.h
lib/Observation.cpp.o: /usr/include/bits/mathdef.h
lib/Observation.cpp.o: /usr/include/bits/mathcalls.h
svm/kernel.cpp.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
svm/kernel.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/kernel.cpp.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
svm/kernel.cpp.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/kernel.cpp.o: /usr/include/stdio.h
svm/kernel.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/kernel.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/kernel.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/kernel.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/kernel.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/kernel.cpp.o: /usr/include/bits/stdio_lim.h lib/Observation.h lib/io.h
svm/Optimizer.cpp.o: lib/common.h
svm/SVM.cpp.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/SVM.cpp.o: /usr/include/gnu/stubs.h
svm/SVM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/SVM.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/SVM.cpp.o: /usr/include/bits/stdio_lim.h lib/io.h lib/Mathmatics.h
svm/SVM.cpp.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM.cpp.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/SVM.cpp.o: /usr/include/ctype.h /usr/include/endian.h
svm/SVM.cpp.o: /usr/include/bits/endian.h /usr/include/string.h
svm/SVM.cpp.o: /usr/include/stdlib.h /usr/include/time.h
svm/SVM.cpp.o: /usr/include/bits/time.h
svm/SVM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
svm/SVM_light.cpp.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.cpp.o: /usr/include/stdio.h /usr/include/features.h
svm/SVM_light.cpp.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/SVM_light.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/SVM_light.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM_light.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM_light.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM_light.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/SVM_light.cpp.o: /usr/include/bits/stdio_lim.h lib/io.h lib/Mathmatics.h
svm/SVM_light.cpp.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM_light.cpp.o: /usr/include/bits/mathdef.h
svm/SVM_light.cpp.o: /usr/include/bits/mathcalls.h /usr/include/ctype.h
svm/SVM_light.cpp.o: /usr/include/endian.h /usr/include/bits/endian.h
svm/SVM_light.cpp.o: /usr/include/string.h /usr/include/stdlib.h
svm/SVM_light.cpp.o: /usr/include/time.h /usr/include/bits/time.h
svm/SVM_light.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
svm/SVM_light.cpp.o: svm/kernel.h hmm/HMM.h svm/Optimizer.h
gui/GeneFinder.cpp.o: lib/io.h /usr/include/stdio.h /usr/include/features.h
gui/GeneFinder.cpp.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
gui/GeneFinder.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
gui/GeneFinder.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
gui/GeneFinder.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
gui/GeneFinder.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
gui/GeneFinder.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
gui/GeneFinder.cpp.o: /usr/include/bits/stdio_lim.h lib/Observation.h
gui/GeneFinder.cpp.o: lib/common.h lib/Mathmatics.h /usr/include/math.h
gui/GeneFinder.cpp.o: /usr/include/bits/huge_val.h
gui/GeneFinder.cpp.o: /usr/include/bits/mathdef.h
gui/GeneFinder.cpp.o: /usr/include/bits/mathcalls.h hmm/HMM.h svm/SVM.h
gui/GeneFinder.cpp.o: /usr/include/ctype.h /usr/include/endian.h
gui/GeneFinder.cpp.o: /usr/include/bits/endian.h /usr/include/string.h
gui/GeneFinder.cpp.o: /usr/include/stdlib.h /usr/include/time.h
gui/GeneFinder.cpp.o: /usr/include/bits/time.h
gui/GeneFinder.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
gui/GeneFinder.cpp.o: svm/SVM_light.h svm/kernel.h svm/Optimizer.h
gui/GeneFinder.cpp.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
gui/GeneFinder.cpp.o: /usr/include/bits/confname.h /usr/include/fcntl.h
gui/GeneFinder.cpp.o: /usr/include/bits/fcntl.h /usr/include/sys/types.h
hmm/HMM.cpp.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
hmm/HMM.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
hmm/HMM.cpp.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
hmm/HMM.cpp.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
hmm/HMM.cpp.o: /usr/include/stdio.h
hmm/HMM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
hmm/HMM.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
hmm/HMM.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
hmm/HMM.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
hmm/HMM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
hmm/HMM.cpp.o: /usr/include/bits/stdio_lim.h lib/Observation.h lib/io.h
hmm/HMM.cpp.o: /usr/include/stdlib.h /usr/include/time.h
hmm/HMM.cpp.o: /usr/include/bits/time.h /usr/include/ctype.h
hmm/HMM.cpp.o: /usr/include/endian.h /usr/include/bits/endian.h
lib/io.cpp.o: lib/io.h /usr/include/stdio.h /usr/include/features.h
lib/io.cpp.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/io.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/io.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
lib/io.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/io.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/io.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/io.cpp.o: /usr/include/bits/stdio_lim.h
lib/Mathmatics.cpp.o: lib/Mathmatics.h lib/common.h /usr/include/math.h
lib/Mathmatics.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
lib/Mathmatics.cpp.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
lib/Mathmatics.cpp.o: /usr/include/bits/mathdef.h
lib/Mathmatics.cpp.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
lib/Mathmatics.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/Mathmatics.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Mathmatics.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Mathmatics.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Mathmatics.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/Mathmatics.cpp.o: /usr/include/bits/stdio_lim.h lib/io.h
lib/Mathmatics.cpp.o: /usr/include/time.h /usr/include/bits/time.h
lib/Mathmatics.cpp.o: /usr/include/stdlib.h
lib/Observation.cpp.o: lib/Observation.h /usr/include/stdio.h
lib/Observation.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
lib/Observation.cpp.o: /usr/include/gnu/stubs.h
lib/Observation.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/Observation.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Observation.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Observation.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Observation.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/Observation.cpp.o: /usr/include/bits/stdio_lim.h lib/io.h lib/common.h
lib/Observation.cpp.o: lib/Mathmatics.h /usr/include/math.h
lib/Observation.cpp.o: /usr/include/bits/huge_val.h
lib/Observation.cpp.o: /usr/include/bits/mathdef.h
lib/Observation.cpp.o: /usr/include/bits/mathcalls.h
svm/kernel.cpp.o: svm/kernel.h hmm/HMM.h lib/Mathmatics.h lib/common.h
svm/kernel.cpp.o: /usr/include/math.h /usr/include/features.h
svm/kernel.cpp.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/kernel.cpp.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
svm/kernel.cpp.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
svm/kernel.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/kernel.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/kernel.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/kernel.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/kernel.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/kernel.cpp.o: /usr/include/bits/stdio_lim.h lib/Observation.h lib/io.h
svm/Optimizer.cpp.o: svm/pr_loqo.h svm/Optimizer.h lib/common.h svm/SVM.h
svm/Optimizer.cpp.o: lib/Observation.h /usr/include/stdio.h
svm/Optimizer.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/Optimizer.cpp.o: /usr/include/gnu/stubs.h
svm/Optimizer.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/Optimizer.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/Optimizer.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/Optimizer.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/Optimizer.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/Optimizer.cpp.o: /usr/include/bits/stdio_lim.h lib/io.h lib/Mathmatics.h
svm/Optimizer.cpp.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/Optimizer.cpp.o: /usr/include/bits/mathdef.h
svm/Optimizer.cpp.o: /usr/include/bits/mathcalls.h /usr/include/ctype.h
svm/Optimizer.cpp.o: /usr/include/endian.h /usr/include/bits/endian.h
svm/Optimizer.cpp.o: /usr/include/string.h /usr/include/stdlib.h
svm/Optimizer.cpp.o: /usr/include/time.h /usr/include/bits/time.h
svm/Optimizer.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
svm/pr_loqo.cpp.o: lib/common.h svm/pr_loqo.h lib/io.h /usr/include/stdio.h
svm/pr_loqo.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/pr_loqo.cpp.o: /usr/include/gnu/stubs.h
svm/pr_loqo.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/pr_loqo.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/pr_loqo.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/pr_loqo.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/pr_loqo.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/pr_loqo.cpp.o: /usr/include/bits/stdio_lim.h /usr/include/math.h
svm/pr_loqo.cpp.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
svm/pr_loqo.cpp.o: /usr/include/bits/mathcalls.h /usr/include/time.h
svm/pr_loqo.cpp.o: /usr/include/bits/time.h /usr/include/stdlib.h
svm/SVM.cpp.o: svm/SVM.h lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.cpp.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/SVM.cpp.o: /usr/include/gnu/stubs.h
svm/SVM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/SVM.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/SVM.cpp.o: /usr/include/bits/stdio_lim.h lib/io.h lib/Mathmatics.h
svm/SVM.cpp.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM.cpp.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/SVM.cpp.o: /usr/include/ctype.h /usr/include/endian.h
svm/SVM.cpp.o: /usr/include/bits/endian.h /usr/include/string.h
svm/SVM.cpp.o: /usr/include/stdlib.h /usr/include/time.h
svm/SVM.cpp.o: /usr/include/bits/time.h
svm/SVM.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
svm/SVM_light.cpp.o: lib/io.h /usr/include/stdio.h /usr/include/features.h
svm/SVM_light.cpp.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/SVM_light.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/SVM_light.cpp.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM_light.cpp.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM_light.cpp.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM_light.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/SVM_light.cpp.o: /usr/include/bits/stdio_lim.h svm/SVM_light.h svm/SVM.h
svm/SVM_light.cpp.o: lib/common.h lib/Observation.h lib/Mathmatics.h
svm/SVM_light.cpp.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM_light.cpp.o: /usr/include/bits/mathdef.h
svm/SVM_light.cpp.o: /usr/include/bits/mathcalls.h /usr/include/ctype.h
svm/SVM_light.cpp.o: /usr/include/endian.h /usr/include/bits/endian.h
svm/SVM_light.cpp.o: /usr/include/string.h /usr/include/stdlib.h
svm/SVM_light.cpp.o: /usr/include/time.h /usr/include/bits/time.h
svm/SVM_light.cpp.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
svm/SVM_light.cpp.o: svm/kernel.h hmm/HMM.h svm/Optimizer.h

hmm/HMM.c.o: lib/Mathmatics.h lib/common.h /usr/include/math.h
hmm/HMM.c.o: /usr/include/features.h /usr/include/sys/cdefs.h
hmm/HMM.c.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
hmm/HMM.c.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
hmm/HMM.c.o: /usr/include/stdio.h
hmm/HMM.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
hmm/HMM.c.o: /usr/include/bits/types.h /usr/include/libio.h
hmm/HMM.c.o: /usr/include/_G_config.h /usr/include/wchar.h
hmm/HMM.c.o: /usr/include/bits/wchar.h /usr/include/gconv.h
hmm/HMM.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
hmm/HMM.c.o: /usr/include/bits/stdio_lim.h lib/Observation.h lib/io.h
lib/io.c.o: /usr/include/stdio.h /usr/include/features.h
lib/io.c.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/io.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/io.c.o: /usr/include/bits/types.h /usr/include/libio.h
lib/io.c.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/io.c.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/io.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/io.c.o: /usr/include/bits/stdio_lim.h
lib/Mathmatics.c.o: lib/common.h /usr/include/math.h /usr/include/features.h
lib/Mathmatics.c.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/Mathmatics.c.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
lib/Mathmatics.c.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
lib/Mathmatics.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/Mathmatics.c.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Mathmatics.c.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Mathmatics.c.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Mathmatics.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/Mathmatics.c.o: /usr/include/bits/stdio_lim.h
lib/Observation.c.o: /usr/include/stdio.h /usr/include/features.h
lib/Observation.c.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/Observation.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
lib/Observation.c.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Observation.c.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Observation.c.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Observation.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
lib/Observation.c.o: /usr/include/bits/stdio_lim.h lib/io.h lib/common.h
lib/Observation.c.o: lib/Mathmatics.h /usr/include/math.h
lib/Observation.c.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
lib/Observation.c.o: /usr/include/bits/mathcalls.h
svm/kernel.c.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
svm/kernel.c.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/kernel.c.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
svm/kernel.c.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/kernel.c.o: /usr/include/stdio.h
svm/kernel.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/kernel.c.o: /usr/include/bits/types.h /usr/include/libio.h
svm/kernel.c.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/kernel.c.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/kernel.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/kernel.c.o: /usr/include/bits/stdio_lim.h lib/Observation.h lib/io.h
svm/Optimizer.c.o: lib/common.h
svm/SVM.c.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.c.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/SVM.c.o: /usr/include/gnu/stubs.h
svm/SVM.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/SVM.c.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM.c.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM.c.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/SVM.c.o: /usr/include/bits/stdio_lim.h lib/io.h lib/Mathmatics.h
svm/SVM.c.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM.c.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/SVM.c.o: /usr/include/ctype.h /usr/include/endian.h
svm/SVM.c.o: /usr/include/bits/endian.h /usr/include/string.h
svm/SVM.c.o: /usr/include/stdlib.h /usr/include/time.h
svm/SVM.c.o: /usr/include/bits/time.h
svm/SVM.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
svm/SVM_light.c.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.c.o: /usr/include/stdio.h /usr/include/features.h
svm/SVM_light.c.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/SVM_light.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stddef.h
svm/SVM_light.c.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM_light.c.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM_light.c.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM_light.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/stdarg.h
svm/SVM_light.c.o: /usr/include/bits/stdio_lim.h lib/io.h lib/Mathmatics.h
svm/SVM_light.c.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM_light.c.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/SVM_light.c.o: /usr/include/ctype.h /usr/include/endian.h
svm/SVM_light.c.o: /usr/include/bits/endian.h /usr/include/string.h
svm/SVM_light.c.o: /usr/include/stdlib.h /usr/include/time.h
svm/SVM_light.c.o: /usr/include/bits/time.h
svm/SVM_light.c.o: /usr/lib/gcc-lib/i386-linux/2.95.4/include/float.h
svm/SVM_light.c.o: svm/kernel.h hmm/HMM.h svm/Optimizer.h
