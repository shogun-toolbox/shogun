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

hmm/HMM.o: /usr/include/math.h /usr/include/sys/cdefs.h /usr/include/stdio.h
hmm/HMM.o: /usr/include/sys/types.h /usr/include/machine/types.h
hmm/HMM.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
hmm/HMM.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
hmm/HMM.o: /usr/include/ppc/endian.h lib/Mathmatics.h /usr/include/stdlib.h
hmm/HMM.o: lib/common.h lib/Observation.h
lib/Mathmatics.o: /usr/include/math.h /usr/include/sys/cdefs.h
lib/Mathmatics.o: /usr/include/stdlib.h /usr/include/machine/ansi.h
lib/Mathmatics.o: /usr/include/ppc/ansi.h lib/common.h
lib/Observation.o: /usr/include/stdio.h /usr/include/sys/types.h
lib/Observation.o: /usr/include/sys/cdefs.h /usr/include/machine/types.h
lib/Observation.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
lib/Observation.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
lib/Observation.o: /usr/include/ppc/endian.h lib/common.h lib/Mathmatics.h
lib/Observation.o: /usr/include/math.h /usr/include/stdlib.h
svm/Optimizer.o: lib/common.h
svm/SVM.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.o: /usr/include/sys/types.h /usr/include/sys/cdefs.h
svm/SVM.o: /usr/include/machine/types.h /usr/include/ppc/types.h
svm/SVM.o: /usr/include/machine/ansi.h /usr/include/ppc/ansi.h
svm/SVM.o: /usr/include/machine/endian.h /usr/include/ppc/endian.h
svm/SVM.o: lib/Mathmatics.h /usr/include/math.h /usr/include/stdlib.h
svm/SVM.o: /usr/include/ctype.h /usr/include/runetype.h /usr/include/string.h
svm/SVM.o: /usr/include/time.h /usr/include/machine/limits.h
svm/SVM.o: /usr/include/ppc/limits.h /usr/include/float.h
svm/SVM_light.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.o: /usr/include/stdio.h /usr/include/sys/types.h
svm/SVM_light.o: /usr/include/sys/cdefs.h /usr/include/machine/types.h
svm/SVM_light.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
svm/SVM_light.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
svm/SVM_light.o: /usr/include/ppc/endian.h lib/Mathmatics.h
svm/SVM_light.o: /usr/include/math.h /usr/include/stdlib.h
svm/SVM_light.o: /usr/include/ctype.h /usr/include/runetype.h
svm/SVM_light.o: /usr/include/string.h /usr/include/time.h
svm/SVM_light.o: /usr/include/machine/limits.h /usr/include/ppc/limits.h
svm/SVM_light.o: /usr/include/float.h svm/kernel.h hmm/HMM.h svm/Optimizer.h
svm/kernel.o: hmm/HMM.h /usr/include/math.h /usr/include/sys/cdefs.h
svm/kernel.o: /usr/include/stdio.h /usr/include/sys/types.h
svm/kernel.o: /usr/include/machine/types.h /usr/include/ppc/types.h
svm/kernel.o: /usr/include/machine/ansi.h /usr/include/ppc/ansi.h
svm/kernel.o: /usr/include/machine/endian.h /usr/include/ppc/endian.h
svm/kernel.o: lib/Mathmatics.h /usr/include/stdlib.h lib/common.h
svm/kernel.o: lib/Observation.h
gui/GeneFinder.o: /usr/include/stdio.h /usr/include/sys/types.h
gui/GeneFinder.o: /usr/include/sys/cdefs.h /usr/include/machine/types.h
gui/GeneFinder.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
gui/GeneFinder.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
gui/GeneFinder.o: /usr/include/ppc/endian.h /usr/include/unistd.h
gui/GeneFinder.o: /usr/include/sys/unistd.h /usr/include/signal.h
gui/GeneFinder.o: /usr/include/sys/signal.h /usr/include/machine/signal.h
gui/GeneFinder.o: /usr/include/ppc/signal.h /usr/include/time.h
gui/GeneFinder.o: /usr/include/machine/limits.h /usr/include/ppc/limits.h
gui/GeneFinder.o: /usr/include/string.h /usr/include/ctype.h
gui/GeneFinder.o: /usr/include/runetype.h /usr/include/stdlib.h
gui/GeneFinder.o: /usr/include/math.h /usr/include/fcntl.h
gui/GeneFinder.o: /usr/include/sys/fcntl.h hmm/HMM.h lib/Mathmatics.h
gui/GeneFinder.o: lib/common.h lib/Observation.h svm/SVM.h
gui/GeneFinder.o: /usr/include/float.h svm/SVM_light.h svm/kernel.h
gui/GeneFinder.o: svm/Optimizer.h
hmm/HMM.o: hmm/HMM.h /usr/include/math.h /usr/include/sys/cdefs.h
hmm/HMM.o: /usr/include/stdio.h /usr/include/sys/types.h
hmm/HMM.o: /usr/include/machine/types.h /usr/include/ppc/types.h
hmm/HMM.o: /usr/include/machine/ansi.h /usr/include/ppc/ansi.h
hmm/HMM.o: /usr/include/machine/endian.h /usr/include/ppc/endian.h
hmm/HMM.o: lib/Mathmatics.h /usr/include/stdlib.h lib/common.h
hmm/HMM.o: lib/Observation.h /usr/include/time.h
hmm/HMM.o: /usr/include/machine/limits.h /usr/include/ppc/limits.h
hmm/HMM.o: /usr/include/ctype.h /usr/include/runetype.h
lib/Mathmatics.o: /usr/include/stdio.h /usr/include/sys/types.h
lib/Mathmatics.o: /usr/include/sys/cdefs.h /usr/include/machine/types.h
lib/Mathmatics.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
lib/Mathmatics.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
lib/Mathmatics.o: /usr/include/ppc/endian.h /usr/include/math.h
lib/Mathmatics.o: /usr/include/time.h /usr/include/machine/limits.h
lib/Mathmatics.o: /usr/include/ppc/limits.h /usr/include/stdlib.h
lib/Mathmatics.o: lib/Mathmatics.h lib/common.h
lib/Observation.o: /usr/include/stdio.h /usr/include/sys/types.h
lib/Observation.o: /usr/include/sys/cdefs.h /usr/include/machine/types.h
lib/Observation.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
lib/Observation.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
lib/Observation.o: /usr/include/ppc/endian.h lib/Observation.h lib/common.h
lib/Observation.o: lib/Mathmatics.h /usr/include/math.h /usr/include/stdlib.h
svm/Optimizer.o: svm/pr_loqo.h svm/Optimizer.h lib/common.h svm/SVM.h
svm/Optimizer.o: lib/Observation.h /usr/include/stdio.h
svm/Optimizer.o: /usr/include/sys/types.h /usr/include/sys/cdefs.h
svm/Optimizer.o: /usr/include/machine/types.h /usr/include/ppc/types.h
svm/Optimizer.o: /usr/include/machine/ansi.h /usr/include/ppc/ansi.h
svm/Optimizer.o: /usr/include/machine/endian.h /usr/include/ppc/endian.h
svm/Optimizer.o: lib/Mathmatics.h /usr/include/math.h /usr/include/stdlib.h
svm/Optimizer.o: /usr/include/ctype.h /usr/include/runetype.h
svm/Optimizer.o: /usr/include/string.h /usr/include/time.h
svm/Optimizer.o: /usr/include/machine/limits.h /usr/include/ppc/limits.h
svm/Optimizer.o: /usr/include/float.h
svm/SVM.o: svm/SVM.h lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.o: /usr/include/sys/types.h /usr/include/sys/cdefs.h
svm/SVM.o: /usr/include/machine/types.h /usr/include/ppc/types.h
svm/SVM.o: /usr/include/machine/ansi.h /usr/include/ppc/ansi.h
svm/SVM.o: /usr/include/machine/endian.h /usr/include/ppc/endian.h
svm/SVM.o: lib/Mathmatics.h /usr/include/math.h /usr/include/stdlib.h
svm/SVM.o: /usr/include/ctype.h /usr/include/runetype.h /usr/include/string.h
svm/SVM.o: /usr/include/time.h /usr/include/machine/limits.h
svm/SVM.o: /usr/include/ppc/limits.h /usr/include/float.h
svm/SVM_light.o: svm/SVM_light.h svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.o: /usr/include/stdio.h /usr/include/sys/types.h
svm/SVM_light.o: /usr/include/sys/cdefs.h /usr/include/machine/types.h
svm/SVM_light.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
svm/SVM_light.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
svm/SVM_light.o: /usr/include/ppc/endian.h lib/Mathmatics.h
svm/SVM_light.o: /usr/include/math.h /usr/include/stdlib.h
svm/SVM_light.o: /usr/include/ctype.h /usr/include/runetype.h
svm/SVM_light.o: /usr/include/string.h /usr/include/time.h
svm/SVM_light.o: /usr/include/machine/limits.h /usr/include/ppc/limits.h
svm/SVM_light.o: /usr/include/float.h svm/kernel.h hmm/HMM.h svm/Optimizer.h
svm/kernel.o: svm/kernel.h hmm/HMM.h /usr/include/math.h
svm/kernel.o: /usr/include/sys/cdefs.h /usr/include/stdio.h
svm/kernel.o: /usr/include/sys/types.h /usr/include/machine/types.h
svm/kernel.o: /usr/include/ppc/types.h /usr/include/machine/ansi.h
svm/kernel.o: /usr/include/ppc/ansi.h /usr/include/machine/endian.h
svm/kernel.o: /usr/include/ppc/endian.h lib/Mathmatics.h
svm/kernel.o: /usr/include/stdlib.h lib/common.h lib/Observation.h
svm/pr_loqo.o: /usr/include/math.h /usr/include/sys/cdefs.h
svm/pr_loqo.o: /usr/include/time.h /usr/include/machine/ansi.h
svm/pr_loqo.o: /usr/include/ppc/ansi.h /usr/include/machine/limits.h
svm/pr_loqo.o: /usr/include/ppc/limits.h /usr/include/stdlib.h
svm/pr_loqo.o: /usr/include/stdio.h /usr/include/sys/types.h
svm/pr_loqo.o: /usr/include/machine/types.h /usr/include/ppc/types.h
svm/pr_loqo.o: /usr/include/machine/endian.h /usr/include/ppc/endian.h
svm/pr_loqo.o: lib/common.h svm/pr_loqo.h
