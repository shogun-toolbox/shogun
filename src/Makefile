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
	$(GDB) genefinder.`uname`

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
hmm/HMM.cpp.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
hmm/HMM.cpp.o: /usr/include/sys/va_list.h lib/Observation.h lib/io.h
hmm/HMM.cpp.o: /usr/include/stdarg.h
lib/Mathmatics.cpp.o: lib/common.h /usr/include/math.h /usr/include/stdio.h
lib/Mathmatics.cpp.o: /usr/include/sys/feature_tests.h
lib/Mathmatics.cpp.o: /usr/include/sys/va_list.h
lib/Observation.cpp.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
lib/Observation.cpp.o: /usr/include/sys/va_list.h lib/io.h
lib/Observation.cpp.o: /usr/include/stdarg.h lib/common.h lib/Mathmatics.h
lib/Observation.cpp.o: /usr/include/math.h
lib/io.cpp.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
lib/io.cpp.o: /usr/include/sys/va_list.h /usr/include/stdarg.h
svm/Optimizer.cpp.o: lib/common.h
svm/SVM.cpp.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.cpp.o: /usr/include/sys/feature_tests.h /usr/include/sys/va_list.h
svm/SVM.cpp.o: lib/io.h /usr/include/stdarg.h lib/Mathmatics.h
svm/SVM.cpp.o: /usr/include/math.h /usr/include/ctype.h /usr/include/string.h
svm/SVM.cpp.o: /usr/include/stdlib.h /usr/include/time.h /usr/include/float.h
svm/SVM_light.cpp.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.cpp.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
svm/SVM_light.cpp.o: /usr/include/sys/va_list.h lib/io.h
svm/SVM_light.cpp.o: /usr/include/stdarg.h lib/Mathmatics.h
svm/SVM_light.cpp.o: /usr/include/math.h /usr/include/ctype.h
svm/SVM_light.cpp.o: /usr/include/string.h /usr/include/stdlib.h
svm/SVM_light.cpp.o: /usr/include/time.h /usr/include/float.h svm/kernel.h
svm/SVM_light.cpp.o: hmm/HMM.h svm/Optimizer.h
svm/kernel.cpp.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
svm/kernel.cpp.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
svm/kernel.cpp.o: /usr/include/sys/va_list.h lib/Observation.h lib/io.h
svm/kernel.cpp.o: /usr/include/stdarg.h
gui/GeneFinder.cpp.o: lib/io.h /usr/include/stdio.h
gui/GeneFinder.cpp.o: /usr/include/sys/feature_tests.h
gui/GeneFinder.cpp.o: /usr/include/sys/va_list.h /usr/include/stdarg.h
gui/GeneFinder.cpp.o: lib/Observation.h lib/common.h lib/Mathmatics.h
gui/GeneFinder.cpp.o: /usr/include/math.h hmm/HMM.h svm/SVM.h
gui/GeneFinder.cpp.o: /usr/include/ctype.h /usr/include/string.h
gui/GeneFinder.cpp.o: /usr/include/stdlib.h /usr/include/time.h
gui/GeneFinder.cpp.o: /usr/include/float.h svm/SVM_light.h svm/kernel.h
gui/GeneFinder.cpp.o: svm/Optimizer.h /usr/include/unistd.h
gui/GeneFinder.cpp.o: /usr/include/sys/types.h /usr/include/sys/isa_defs.h
gui/GeneFinder.cpp.o: /usr/include/sys/machtypes.h /usr/include/sys/select.h
gui/GeneFinder.cpp.o: /usr/include/sys/time.h /usr/include/sys/unistd.h
gui/GeneFinder.cpp.o: /usr/include/fcntl.h /usr/include/sys/fcntl.h
hmm/HMM.cpp.o: /usr/include/math.h hmm/HMM.h lib/Mathmatics.h lib/common.h
hmm/HMM.cpp.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
hmm/HMM.cpp.o: /usr/include/sys/va_list.h lib/Observation.h lib/io.h
hmm/HMM.cpp.o: /usr/include/stdarg.h /usr/include/stdlib.h
hmm/HMM.cpp.o: /usr/include/time.h /usr/include/ctype.h
lib/Mathmatics.cpp.o: lib/Mathmatics.h lib/common.h /usr/include/math.h
lib/Mathmatics.cpp.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
lib/Mathmatics.cpp.o: /usr/include/sys/va_list.h lib/io.h
lib/Mathmatics.cpp.o: /usr/include/stdarg.h /usr/include/time.h
lib/Mathmatics.cpp.o: /usr/include/stdlib.h
lib/Observation.cpp.o: lib/Observation.h /usr/include/stdio.h
lib/Observation.cpp.o: /usr/include/sys/feature_tests.h
lib/Observation.cpp.o: /usr/include/sys/va_list.h lib/io.h
lib/Observation.cpp.o: /usr/include/stdarg.h lib/common.h lib/Mathmatics.h
lib/Observation.cpp.o: /usr/include/math.h
lib/io.cpp.o: lib/io.h /usr/include/stdio.h /usr/include/sys/feature_tests.h
lib/io.cpp.o: /usr/include/sys/va_list.h /usr/include/stdarg.h
svm/Optimizer.cpp.o: svm/pr_loqo.h svm/Optimizer.h lib/common.h svm/SVM.h
svm/Optimizer.cpp.o: lib/Observation.h /usr/include/stdio.h
svm/Optimizer.cpp.o: /usr/include/sys/feature_tests.h
svm/Optimizer.cpp.o: /usr/include/sys/va_list.h lib/io.h
svm/Optimizer.cpp.o: /usr/include/stdarg.h lib/Mathmatics.h
svm/Optimizer.cpp.o: /usr/include/math.h /usr/include/ctype.h
svm/Optimizer.cpp.o: /usr/include/string.h /usr/include/stdlib.h
svm/Optimizer.cpp.o: /usr/include/time.h /usr/include/float.h
svm/SVM.cpp.o: svm/SVM.h lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.cpp.o: /usr/include/sys/feature_tests.h /usr/include/sys/va_list.h
svm/SVM.cpp.o: lib/io.h /usr/include/stdarg.h lib/Mathmatics.h
svm/SVM.cpp.o: /usr/include/math.h /usr/include/ctype.h /usr/include/string.h
svm/SVM.cpp.o: /usr/include/stdlib.h /usr/include/time.h /usr/include/float.h
svm/SVM_light.cpp.o: lib/io.h /usr/include/stdio.h
svm/SVM_light.cpp.o: /usr/include/sys/feature_tests.h
svm/SVM_light.cpp.o: /usr/include/sys/va_list.h /usr/include/stdarg.h
svm/SVM_light.cpp.o: svm/SVM_light.h svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.cpp.o: lib/Mathmatics.h /usr/include/math.h
svm/SVM_light.cpp.o: /usr/include/ctype.h /usr/include/string.h
svm/SVM_light.cpp.o: /usr/include/stdlib.h /usr/include/time.h
svm/SVM_light.cpp.o: /usr/include/float.h svm/kernel.h hmm/HMM.h
svm/SVM_light.cpp.o: svm/Optimizer.h
svm/kernel.cpp.o: svm/kernel.h hmm/HMM.h lib/Mathmatics.h lib/common.h
svm/kernel.cpp.o: /usr/include/math.h /usr/include/stdio.h
svm/kernel.cpp.o: /usr/include/sys/feature_tests.h /usr/include/sys/va_list.h
svm/kernel.cpp.o: lib/Observation.h lib/io.h /usr/include/stdarg.h
svm/pr_loqo.cpp.o: lib/common.h svm/pr_loqo.h lib/io.h /usr/include/stdio.h
svm/pr_loqo.cpp.o: /usr/include/sys/feature_tests.h
svm/pr_loqo.cpp.o: /usr/include/sys/va_list.h /usr/include/stdarg.h
svm/pr_loqo.cpp.o: /usr/include/math.h /usr/include/time.h
svm/pr_loqo.cpp.o: /usr/include/stdlib.h

hmm/HMM.c.o: lib/Mathmatics.h lib/common.h /usr/include/math.h
hmm/HMM.c.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
hmm/HMM.c.o: /usr/include/sys/va_list.h lib/Observation.h lib/io.h
hmm/HMM.c.o: /usr/include/stdarg.h
lib/Mathmatics.c.o: lib/common.h /usr/include/math.h /usr/include/stdio.h
lib/Mathmatics.c.o: /usr/include/sys/feature_tests.h
lib/Mathmatics.c.o: /usr/include/sys/va_list.h
lib/Observation.c.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
lib/Observation.c.o: /usr/include/sys/va_list.h lib/io.h
lib/Observation.c.o: /usr/include/stdarg.h lib/common.h lib/Mathmatics.h
lib/Observation.c.o: /usr/include/math.h
lib/io.c.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
lib/io.c.o: /usr/include/sys/va_list.h /usr/include/stdarg.h
svm/Optimizer.c.o: lib/common.h
svm/SVM.c.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.c.o: /usr/include/sys/feature_tests.h /usr/include/sys/va_list.h
svm/SVM.c.o: lib/io.h /usr/include/stdarg.h lib/Mathmatics.h
svm/SVM.c.o: /usr/include/math.h /usr/include/ctype.h /usr/include/string.h
svm/SVM.c.o: /usr/include/stdlib.h /usr/include/time.h /usr/include/float.h
svm/SVM_light.c.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.c.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
svm/SVM_light.c.o: /usr/include/sys/va_list.h lib/io.h /usr/include/stdarg.h
svm/SVM_light.c.o: lib/Mathmatics.h /usr/include/math.h /usr/include/ctype.h
svm/SVM_light.c.o: /usr/include/string.h /usr/include/stdlib.h
svm/SVM_light.c.o: /usr/include/time.h /usr/include/float.h svm/kernel.h
svm/SVM_light.c.o: hmm/HMM.h svm/Optimizer.h
svm/kernel.c.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
svm/kernel.c.o: /usr/include/stdio.h /usr/include/sys/feature_tests.h
svm/kernel.c.o: /usr/include/sys/va_list.h lib/Observation.h lib/io.h
svm/kernel.c.o: /usr/include/stdarg.h
