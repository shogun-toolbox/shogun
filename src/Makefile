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

hmm/HMM.o: /usr/include/math.h /usr/include/standards.h /usr/include/stdlib.h
hmm/HMM.o: /usr/include/getopt.h /usr/include/sys/types.h
hmm/HMM.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
hmm/HMM.o: /usr/include/sys/select.h /usr/include/stdio.h
hmm/HMM.o: /usr/include/sys/seek.h /usr/include/va_list.h
hmm/HMM.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
hmm/HMM.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
hmm/HMM.o: /usr/include/sys/rt_limits.h lib/Mathmatics.h lib/common.h
hmm/HMM.o: lib/Observation.h
lib/Mathmatics.o: /usr/include/math.h /usr/include/standards.h
lib/Mathmatics.o: /usr/include/stdlib.h /usr/include/getopt.h
lib/Mathmatics.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/Mathmatics.o: /usr/include/mach/machine/vm_types.h
lib/Mathmatics.o: /usr/include/sys/select.h lib/common.h
lib/Observation.o: /usr/include/stdio.h /usr/include/standards.h
lib/Observation.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Observation.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/Observation.o: /usr/include/mach/machine/vm_types.h
lib/Observation.o: /usr/include/sys/select.h /usr/include/getopt.h
lib/Observation.o: /usr/include/sys/limits.h
lib/Observation.o: /usr/include/machine/machlimits.h
lib/Observation.o: /usr/include/sys/syslimits.h
lib/Observation.o: /usr/include/machine/machtime.h
lib/Observation.o: /usr/include/sys/rt_limits.h lib/common.h lib/Mathmatics.h
lib/Observation.o: /usr/include/math.h /usr/include/stdlib.h
svm/Optimizer.o: lib/common.h
svm/SVM.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/SVM.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/SVM.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/SVM.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/SVM.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/SVM.o: /usr/include/sys/rt_limits.h lib/Mathmatics.h /usr/include/math.h
svm/SVM.o: /usr/include/stdlib.h /usr/include/ctype.h
svm/SVM.o: /usr/include/sys/localedef.h /usr/include/sys/lc_core.h
svm/SVM.o: /usr/include/locale.h /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM.o: /usr/include/limits.h /usr/include/string.h /usr/include/strings.h
svm/SVM.o: /usr/include/time.h /usr/include/sys/timers.h
svm/SVM.o: /usr/include/sys/signalmisc.h /usr/include/sys/timemisc.h
svm/SVM.o: /usr/include/float.h /usr/include/fp_class.h
svm/SVM_light.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.o: /usr/include/stdio.h /usr/include/standards.h
svm/SVM_light.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/SVM_light.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/SVM_light.o: /usr/include/mach/machine/vm_types.h
svm/SVM_light.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM_light.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/SVM_light.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/SVM_light.o: /usr/include/sys/rt_limits.h lib/Mathmatics.h
svm/SVM_light.o: /usr/include/math.h /usr/include/stdlib.h
svm/SVM_light.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/SVM_light.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/SVM_light.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM_light.o: /usr/include/limits.h /usr/include/string.h
svm/SVM_light.o: /usr/include/strings.h /usr/include/time.h
svm/SVM_light.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/SVM_light.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/SVM_light.o: /usr/include/fp_class.h svm/kernel.h hmm/HMM.h
svm/SVM_light.o: svm/Optimizer.h
svm/kernel.o: hmm/HMM.h /usr/include/math.h /usr/include/standards.h
svm/kernel.o: /usr/include/stdlib.h /usr/include/getopt.h
svm/kernel.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/kernel.o: /usr/include/mach/machine/vm_types.h /usr/include/sys/select.h
svm/kernel.o: /usr/include/stdio.h /usr/include/sys/seek.h
svm/kernel.o: /usr/include/va_list.h /usr/include/sys/limits.h
svm/kernel.o: /usr/include/machine/machlimits.h /usr/include/sys/syslimits.h
svm/kernel.o: /usr/include/machine/machtime.h /usr/include/sys/rt_limits.h
svm/kernel.o: lib/Mathmatics.h lib/common.h lib/Observation.h
gui/GeneFinder.o: /usr/include/stdio.h /usr/include/standards.h
gui/GeneFinder.o: /usr/include/sys/seek.h /usr/include/va_list.h
gui/GeneFinder.o: /usr/include/sys/types.h /usr/include/time_t.h
gui/GeneFinder.o: /usr/include/mach/machine/vm_types.h
gui/GeneFinder.o: /usr/include/sys/select.h /usr/include/getopt.h
gui/GeneFinder.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
gui/GeneFinder.o: /usr/include/sys/syslimits.h
gui/GeneFinder.o: /usr/include/machine/machtime.h
gui/GeneFinder.o: /usr/include/sys/rt_limits.h /usr/include/unistd.h
gui/GeneFinder.o: /usr/include/sys/access.h /usr/include/time.h
gui/GeneFinder.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
gui/GeneFinder.o: /usr/include/sys/timemisc.h /usr/include/string.h
gui/GeneFinder.o: /usr/include/strings.h /usr/include/ctype.h
gui/GeneFinder.o: /usr/include/sys/localedef.h /usr/include/sys/lc_core.h
gui/GeneFinder.o: /usr/include/locale.h /usr/include/nl_types.h
gui/GeneFinder.o: /usr/include/mesg.h /usr/include/limits.h
gui/GeneFinder.o: /usr/include/stdlib.h /usr/include/math.h
gui/GeneFinder.o: /usr/include/fcntl.h /usr/include/sys/fcntl.h
gui/GeneFinder.o: /usr/include/sys/mode.h hmm/HMM.h lib/Mathmatics.h
gui/GeneFinder.o: lib/common.h lib/Observation.h svm/SVM.h
gui/GeneFinder.o: /usr/include/float.h /usr/include/fp_class.h
gui/GeneFinder.o: svm/SVM_light.h svm/kernel.h svm/Optimizer.h
hmm/HMM.o: hmm/HMM.h /usr/include/math.h /usr/include/standards.h
hmm/HMM.o: /usr/include/stdlib.h /usr/include/getopt.h
hmm/HMM.o: /usr/include/sys/types.h /usr/include/time_t.h
hmm/HMM.o: /usr/include/mach/machine/vm_types.h /usr/include/sys/select.h
hmm/HMM.o: /usr/include/stdio.h /usr/include/sys/seek.h
hmm/HMM.o: /usr/include/va_list.h /usr/include/sys/limits.h
hmm/HMM.o: /usr/include/machine/machlimits.h /usr/include/sys/syslimits.h
hmm/HMM.o: /usr/include/machine/machtime.h /usr/include/sys/rt_limits.h
hmm/HMM.o: lib/Mathmatics.h lib/common.h lib/Observation.h
hmm/HMM.o: /usr/include/time.h /usr/include/sys/timers.h
hmm/HMM.o: /usr/include/sys/signalmisc.h /usr/include/sys/timemisc.h
hmm/HMM.o: /usr/include/ctype.h /usr/include/sys/localedef.h
hmm/HMM.o: /usr/include/sys/lc_core.h /usr/include/locale.h
hmm/HMM.o: /usr/include/nl_types.h /usr/include/mesg.h /usr/include/limits.h
lib/Mathmatics.o: /usr/include/stdio.h /usr/include/standards.h
lib/Mathmatics.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Mathmatics.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/Mathmatics.o: /usr/include/mach/machine/vm_types.h
lib/Mathmatics.o: /usr/include/sys/select.h /usr/include/getopt.h
lib/Mathmatics.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
lib/Mathmatics.o: /usr/include/sys/syslimits.h
lib/Mathmatics.o: /usr/include/machine/machtime.h
lib/Mathmatics.o: /usr/include/sys/rt_limits.h /usr/include/math.h
lib/Mathmatics.o: /usr/include/stdlib.h /usr/include/time.h
lib/Mathmatics.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
lib/Mathmatics.o: /usr/include/sys/timemisc.h lib/Mathmatics.h lib/common.h
lib/Observation.o: /usr/include/stdio.h /usr/include/standards.h
lib/Observation.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Observation.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/Observation.o: /usr/include/mach/machine/vm_types.h
lib/Observation.o: /usr/include/sys/select.h /usr/include/getopt.h
lib/Observation.o: /usr/include/sys/limits.h
lib/Observation.o: /usr/include/machine/machlimits.h
lib/Observation.o: /usr/include/sys/syslimits.h
lib/Observation.o: /usr/include/machine/machtime.h
lib/Observation.o: /usr/include/sys/rt_limits.h lib/Observation.h
lib/Observation.o: lib/common.h lib/Mathmatics.h /usr/include/math.h
lib/Observation.o: /usr/include/stdlib.h
svm/Optimizer.o: svm/pr_loqo.h svm/Optimizer.h lib/common.h svm/SVM.h
svm/Optimizer.o: lib/Observation.h /usr/include/stdio.h
svm/Optimizer.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/Optimizer.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/Optimizer.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/Optimizer.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/Optimizer.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/Optimizer.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/Optimizer.o: /usr/include/sys/rt_limits.h lib/Mathmatics.h
svm/Optimizer.o: /usr/include/math.h /usr/include/stdlib.h
svm/Optimizer.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/Optimizer.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/Optimizer.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/Optimizer.o: /usr/include/limits.h /usr/include/string.h
svm/Optimizer.o: /usr/include/strings.h /usr/include/time.h
svm/Optimizer.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/Optimizer.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/Optimizer.o: /usr/include/fp_class.h
svm/SVM.o: svm/SVM.h lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/SVM.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/SVM.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/SVM.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/SVM.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/SVM.o: /usr/include/sys/rt_limits.h lib/Mathmatics.h /usr/include/math.h
svm/SVM.o: /usr/include/stdlib.h /usr/include/ctype.h
svm/SVM.o: /usr/include/sys/localedef.h /usr/include/sys/lc_core.h
svm/SVM.o: /usr/include/locale.h /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM.o: /usr/include/limits.h /usr/include/string.h /usr/include/strings.h
svm/SVM.o: /usr/include/time.h /usr/include/sys/timers.h
svm/SVM.o: /usr/include/sys/signalmisc.h /usr/include/sys/timemisc.h
svm/SVM.o: /usr/include/float.h /usr/include/fp_class.h
svm/SVM_light.o: svm/SVM_light.h svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.o: /usr/include/stdio.h /usr/include/standards.h
svm/SVM_light.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/SVM_light.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/SVM_light.o: /usr/include/mach/machine/vm_types.h
svm/SVM_light.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM_light.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/SVM_light.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/SVM_light.o: /usr/include/sys/rt_limits.h lib/Mathmatics.h
svm/SVM_light.o: /usr/include/math.h /usr/include/stdlib.h
svm/SVM_light.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/SVM_light.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/SVM_light.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM_light.o: /usr/include/limits.h /usr/include/string.h
svm/SVM_light.o: /usr/include/strings.h /usr/include/time.h
svm/SVM_light.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/SVM_light.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/SVM_light.o: /usr/include/fp_class.h svm/kernel.h hmm/HMM.h
svm/SVM_light.o: svm/Optimizer.h
svm/kernel.o: svm/kernel.h hmm/HMM.h /usr/include/math.h
svm/kernel.o: /usr/include/standards.h /usr/include/stdlib.h
svm/kernel.o: /usr/include/getopt.h /usr/include/sys/types.h
svm/kernel.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/kernel.o: /usr/include/sys/select.h /usr/include/stdio.h
svm/kernel.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/kernel.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/kernel.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/kernel.o: /usr/include/sys/rt_limits.h lib/Mathmatics.h lib/common.h
svm/kernel.o: lib/Observation.h
svm/pr_loqo.o: /usr/include/math.h /usr/include/standards.h
svm/pr_loqo.o: /usr/include/stdlib.h /usr/include/getopt.h
svm/pr_loqo.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/pr_loqo.o: /usr/include/mach/machine/vm_types.h /usr/include/sys/select.h
svm/pr_loqo.o: /usr/include/time.h /usr/include/machine/machtime.h
svm/pr_loqo.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/pr_loqo.o: /usr/include/sys/timemisc.h /usr/include/stdio.h
svm/pr_loqo.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/pr_loqo.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/pr_loqo.o: /usr/include/sys/syslimits.h /usr/include/sys/rt_limits.h
svm/pr_loqo.o: lib/common.h svm/pr_loqo.h
