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
hmm/HMM.cpp.o: /usr/include/standards.h /usr/include/stdlib.h
hmm/HMM.cpp.o: /usr/include/getopt.h /usr/include/sys/types.h
hmm/HMM.cpp.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
hmm/HMM.cpp.o: /usr/include/sys/select.h /usr/include/stdio.h
hmm/HMM.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
hmm/HMM.cpp.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
hmm/HMM.cpp.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
hmm/HMM.cpp.o: /usr/include/sys/rt_limits.h lib/Observation.h lib/io.h
hmm/HMM.cpp.o: /usr/include/stdarg.h
lib/Mathmatics.cpp.o: lib/common.h /usr/include/math.h
lib/Mathmatics.cpp.o: /usr/include/standards.h /usr/include/stdlib.h
lib/Mathmatics.cpp.o: /usr/include/getopt.h /usr/include/sys/types.h
lib/Mathmatics.cpp.o: /usr/include/time_t.h
lib/Mathmatics.cpp.o: /usr/include/mach/machine/vm_types.h
lib/Mathmatics.cpp.o: /usr/include/sys/select.h /usr/include/stdio.h
lib/Mathmatics.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Mathmatics.cpp.o: /usr/include/sys/limits.h
lib/Mathmatics.cpp.o: /usr/include/machine/machlimits.h
lib/Mathmatics.cpp.o: /usr/include/sys/syslimits.h
lib/Mathmatics.cpp.o: /usr/include/machine/machtime.h
lib/Mathmatics.cpp.o: /usr/include/sys/rt_limits.h
lib/Observation.cpp.o: /usr/include/stdio.h /usr/include/standards.h
lib/Observation.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Observation.cpp.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/Observation.cpp.o: /usr/include/mach/machine/vm_types.h
lib/Observation.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
lib/Observation.cpp.o: /usr/include/sys/limits.h
lib/Observation.cpp.o: /usr/include/machine/machlimits.h
lib/Observation.cpp.o: /usr/include/sys/syslimits.h
lib/Observation.cpp.o: /usr/include/machine/machtime.h
lib/Observation.cpp.o: /usr/include/sys/rt_limits.h lib/io.h
lib/Observation.cpp.o: /usr/include/stdarg.h lib/common.h lib/Mathmatics.h
lib/Observation.cpp.o: /usr/include/math.h /usr/include/stdlib.h
lib/io.cpp.o: /usr/include/stdio.h /usr/include/standards.h
lib/io.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/io.cpp.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/io.cpp.o: /usr/include/mach/machine/vm_types.h /usr/include/sys/select.h
lib/io.cpp.o: /usr/include/getopt.h /usr/include/sys/limits.h
lib/io.cpp.o: /usr/include/machine/machlimits.h /usr/include/sys/syslimits.h
lib/io.cpp.o: /usr/include/machine/machtime.h /usr/include/sys/rt_limits.h
lib/io.cpp.o: /usr/include/stdarg.h
svm/Optimizer.cpp.o: lib/common.h
svm/SVM.cpp.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.cpp.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/SVM.cpp.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/SVM.cpp.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/SVM.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM.cpp.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/SVM.cpp.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/SVM.cpp.o: /usr/include/sys/rt_limits.h lib/io.h /usr/include/stdarg.h
svm/SVM.cpp.o: lib/Mathmatics.h /usr/include/math.h /usr/include/stdlib.h
svm/SVM.cpp.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/SVM.cpp.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/SVM.cpp.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM.cpp.o: /usr/include/limits.h /usr/include/string.h
svm/SVM.cpp.o: /usr/include/strings.h /usr/include/time.h
svm/SVM.cpp.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/SVM.cpp.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/SVM.cpp.o: /usr/include/fp_class.h
svm/SVM_light.cpp.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.cpp.o: /usr/include/stdio.h /usr/include/standards.h
svm/SVM_light.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/SVM_light.cpp.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/SVM_light.cpp.o: /usr/include/mach/machine/vm_types.h
svm/SVM_light.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM_light.cpp.o: /usr/include/sys/limits.h
svm/SVM_light.cpp.o: /usr/include/machine/machlimits.h
svm/SVM_light.cpp.o: /usr/include/sys/syslimits.h
svm/SVM_light.cpp.o: /usr/include/machine/machtime.h
svm/SVM_light.cpp.o: /usr/include/sys/rt_limits.h lib/io.h
svm/SVM_light.cpp.o: /usr/include/stdarg.h lib/Mathmatics.h
svm/SVM_light.cpp.o: /usr/include/math.h /usr/include/stdlib.h
svm/SVM_light.cpp.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/SVM_light.cpp.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/SVM_light.cpp.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM_light.cpp.o: /usr/include/limits.h /usr/include/string.h
svm/SVM_light.cpp.o: /usr/include/strings.h /usr/include/time.h
svm/SVM_light.cpp.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/SVM_light.cpp.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/SVM_light.cpp.o: /usr/include/fp_class.h svm/kernel.h hmm/HMM.h
svm/SVM_light.cpp.o: svm/Optimizer.h
svm/kernel.cpp.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
svm/kernel.cpp.o: /usr/include/standards.h /usr/include/stdlib.h
svm/kernel.cpp.o: /usr/include/getopt.h /usr/include/sys/types.h
svm/kernel.cpp.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/kernel.cpp.o: /usr/include/sys/select.h /usr/include/stdio.h
svm/kernel.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/kernel.cpp.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/kernel.cpp.o: /usr/include/sys/syslimits.h
svm/kernel.cpp.o: /usr/include/machine/machtime.h
svm/kernel.cpp.o: /usr/include/sys/rt_limits.h lib/Observation.h lib/io.h
svm/kernel.cpp.o: /usr/include/stdarg.h
gui/GeneFinder.cpp.o: lib/io.h /usr/include/stdio.h /usr/include/standards.h
gui/GeneFinder.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
gui/GeneFinder.cpp.o: /usr/include/sys/types.h /usr/include/time_t.h
gui/GeneFinder.cpp.o: /usr/include/mach/machine/vm_types.h
gui/GeneFinder.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
gui/GeneFinder.cpp.o: /usr/include/sys/limits.h
gui/GeneFinder.cpp.o: /usr/include/machine/machlimits.h
gui/GeneFinder.cpp.o: /usr/include/sys/syslimits.h
gui/GeneFinder.cpp.o: /usr/include/machine/machtime.h
gui/GeneFinder.cpp.o: /usr/include/sys/rt_limits.h /usr/include/stdarg.h
gui/GeneFinder.cpp.o: lib/Observation.h lib/common.h lib/Mathmatics.h
gui/GeneFinder.cpp.o: /usr/include/math.h /usr/include/stdlib.h hmm/HMM.h
gui/GeneFinder.cpp.o: svm/SVM.h /usr/include/ctype.h
gui/GeneFinder.cpp.o: /usr/include/sys/localedef.h /usr/include/sys/lc_core.h
gui/GeneFinder.cpp.o: /usr/include/locale.h /usr/include/nl_types.h
gui/GeneFinder.cpp.o: /usr/include/mesg.h /usr/include/limits.h
gui/GeneFinder.cpp.o: /usr/include/string.h /usr/include/strings.h
gui/GeneFinder.cpp.o: /usr/include/time.h /usr/include/sys/timers.h
gui/GeneFinder.cpp.o: /usr/include/sys/signalmisc.h
gui/GeneFinder.cpp.o: /usr/include/sys/timemisc.h /usr/include/float.h
gui/GeneFinder.cpp.o: /usr/include/fp_class.h svm/SVM_light.h svm/kernel.h
gui/GeneFinder.cpp.o: svm/Optimizer.h /usr/include/unistd.h
gui/GeneFinder.cpp.o: /usr/include/sys/access.h /usr/include/fcntl.h
gui/GeneFinder.cpp.o: /usr/include/sys/fcntl.h /usr/include/sys/mode.h
hmm/HMM.cpp.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
hmm/HMM.cpp.o: /usr/include/standards.h /usr/include/stdlib.h
hmm/HMM.cpp.o: /usr/include/getopt.h /usr/include/sys/types.h
hmm/HMM.cpp.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
hmm/HMM.cpp.o: /usr/include/sys/select.h /usr/include/stdio.h
hmm/HMM.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
hmm/HMM.cpp.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
hmm/HMM.cpp.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
hmm/HMM.cpp.o: /usr/include/sys/rt_limits.h lib/Observation.h lib/io.h
hmm/HMM.cpp.o: /usr/include/stdarg.h /usr/include/time.h
hmm/HMM.cpp.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
hmm/HMM.cpp.o: /usr/include/sys/timemisc.h /usr/include/ctype.h
hmm/HMM.cpp.o: /usr/include/sys/localedef.h /usr/include/sys/lc_core.h
hmm/HMM.cpp.o: /usr/include/locale.h /usr/include/nl_types.h
hmm/HMM.cpp.o: /usr/include/mesg.h /usr/include/limits.h
lib/Mathmatics.cpp.o: lib/Mathmatics.h lib/common.h /usr/include/math.h
lib/Mathmatics.cpp.o: /usr/include/standards.h /usr/include/stdlib.h
lib/Mathmatics.cpp.o: /usr/include/getopt.h /usr/include/sys/types.h
lib/Mathmatics.cpp.o: /usr/include/time_t.h
lib/Mathmatics.cpp.o: /usr/include/mach/machine/vm_types.h
lib/Mathmatics.cpp.o: /usr/include/sys/select.h /usr/include/stdio.h
lib/Mathmatics.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Mathmatics.cpp.o: /usr/include/sys/limits.h
lib/Mathmatics.cpp.o: /usr/include/machine/machlimits.h
lib/Mathmatics.cpp.o: /usr/include/sys/syslimits.h
lib/Mathmatics.cpp.o: /usr/include/machine/machtime.h
lib/Mathmatics.cpp.o: /usr/include/sys/rt_limits.h lib/io.h
lib/Mathmatics.cpp.o: /usr/include/stdarg.h /usr/include/time.h
lib/Mathmatics.cpp.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
lib/Mathmatics.cpp.o: /usr/include/sys/timemisc.h
lib/Observation.cpp.o: lib/Observation.h /usr/include/stdio.h
lib/Observation.cpp.o: /usr/include/standards.h /usr/include/sys/seek.h
lib/Observation.cpp.o: /usr/include/va_list.h /usr/include/sys/types.h
lib/Observation.cpp.o: /usr/include/time_t.h
lib/Observation.cpp.o: /usr/include/mach/machine/vm_types.h
lib/Observation.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
lib/Observation.cpp.o: /usr/include/sys/limits.h
lib/Observation.cpp.o: /usr/include/machine/machlimits.h
lib/Observation.cpp.o: /usr/include/sys/syslimits.h
lib/Observation.cpp.o: /usr/include/machine/machtime.h
lib/Observation.cpp.o: /usr/include/sys/rt_limits.h lib/io.h
lib/Observation.cpp.o: /usr/include/stdarg.h lib/common.h lib/Mathmatics.h
lib/Observation.cpp.o: /usr/include/math.h /usr/include/stdlib.h
lib/io.cpp.o: lib/io.h /usr/include/stdio.h /usr/include/standards.h
lib/io.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/io.cpp.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/io.cpp.o: /usr/include/mach/machine/vm_types.h /usr/include/sys/select.h
lib/io.cpp.o: /usr/include/getopt.h /usr/include/sys/limits.h
lib/io.cpp.o: /usr/include/machine/machlimits.h /usr/include/sys/syslimits.h
lib/io.cpp.o: /usr/include/machine/machtime.h /usr/include/sys/rt_limits.h
lib/io.cpp.o: /usr/include/stdarg.h
svm/Optimizer.cpp.o: svm/pr_loqo.h svm/Optimizer.h lib/common.h svm/SVM.h
svm/Optimizer.cpp.o: lib/Observation.h /usr/include/stdio.h
svm/Optimizer.cpp.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/Optimizer.cpp.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/Optimizer.cpp.o: /usr/include/time_t.h
svm/Optimizer.cpp.o: /usr/include/mach/machine/vm_types.h
svm/Optimizer.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/Optimizer.cpp.o: /usr/include/sys/limits.h
svm/Optimizer.cpp.o: /usr/include/machine/machlimits.h
svm/Optimizer.cpp.o: /usr/include/sys/syslimits.h
svm/Optimizer.cpp.o: /usr/include/machine/machtime.h
svm/Optimizer.cpp.o: /usr/include/sys/rt_limits.h lib/io.h
svm/Optimizer.cpp.o: /usr/include/stdarg.h lib/Mathmatics.h
svm/Optimizer.cpp.o: /usr/include/math.h /usr/include/stdlib.h
svm/Optimizer.cpp.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/Optimizer.cpp.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/Optimizer.cpp.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/Optimizer.cpp.o: /usr/include/limits.h /usr/include/string.h
svm/Optimizer.cpp.o: /usr/include/strings.h /usr/include/time.h
svm/Optimizer.cpp.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/Optimizer.cpp.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/Optimizer.cpp.o: /usr/include/fp_class.h
svm/SVM.cpp.o: svm/SVM.h lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.cpp.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/SVM.cpp.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/SVM.cpp.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/SVM.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM.cpp.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/SVM.cpp.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/SVM.cpp.o: /usr/include/sys/rt_limits.h lib/io.h /usr/include/stdarg.h
svm/SVM.cpp.o: lib/Mathmatics.h /usr/include/math.h /usr/include/stdlib.h
svm/SVM.cpp.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/SVM.cpp.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/SVM.cpp.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM.cpp.o: /usr/include/limits.h /usr/include/string.h
svm/SVM.cpp.o: /usr/include/strings.h /usr/include/time.h
svm/SVM.cpp.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/SVM.cpp.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/SVM.cpp.o: /usr/include/fp_class.h
svm/SVM_light.cpp.o: lib/io.h /usr/include/stdio.h /usr/include/standards.h
svm/SVM_light.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/SVM_light.cpp.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/SVM_light.cpp.o: /usr/include/mach/machine/vm_types.h
svm/SVM_light.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM_light.cpp.o: /usr/include/sys/limits.h
svm/SVM_light.cpp.o: /usr/include/machine/machlimits.h
svm/SVM_light.cpp.o: /usr/include/sys/syslimits.h
svm/SVM_light.cpp.o: /usr/include/machine/machtime.h
svm/SVM_light.cpp.o: /usr/include/sys/rt_limits.h /usr/include/stdarg.h
svm/SVM_light.cpp.o: svm/SVM_light.h svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.cpp.o: lib/Mathmatics.h /usr/include/math.h
svm/SVM_light.cpp.o: /usr/include/stdlib.h /usr/include/ctype.h
svm/SVM_light.cpp.o: /usr/include/sys/localedef.h /usr/include/sys/lc_core.h
svm/SVM_light.cpp.o: /usr/include/locale.h /usr/include/nl_types.h
svm/SVM_light.cpp.o: /usr/include/mesg.h /usr/include/limits.h
svm/SVM_light.cpp.o: /usr/include/string.h /usr/include/strings.h
svm/SVM_light.cpp.o: /usr/include/time.h /usr/include/sys/timers.h
svm/SVM_light.cpp.o: /usr/include/sys/signalmisc.h
svm/SVM_light.cpp.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/SVM_light.cpp.o: /usr/include/fp_class.h svm/kernel.h hmm/HMM.h
svm/SVM_light.cpp.o: svm/Optimizer.h
svm/kernel.cpp.o: svm/kernel.h hmm/HMM.h lib/Mathmatics.h lib/common.h
svm/kernel.cpp.o: /usr/include/math.h /usr/include/standards.h
svm/kernel.cpp.o: /usr/include/stdlib.h /usr/include/getopt.h
svm/kernel.cpp.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/kernel.cpp.o: /usr/include/mach/machine/vm_types.h
svm/kernel.cpp.o: /usr/include/sys/select.h /usr/include/stdio.h
svm/kernel.cpp.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/kernel.cpp.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/kernel.cpp.o: /usr/include/sys/syslimits.h
svm/kernel.cpp.o: /usr/include/machine/machtime.h
svm/kernel.cpp.o: /usr/include/sys/rt_limits.h lib/Observation.h lib/io.h
svm/kernel.cpp.o: /usr/include/stdarg.h
svm/pr_loqo.cpp.o: lib/common.h svm/pr_loqo.h lib/io.h /usr/include/stdio.h
svm/pr_loqo.cpp.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/pr_loqo.cpp.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/pr_loqo.cpp.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/pr_loqo.cpp.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/pr_loqo.cpp.o: /usr/include/sys/limits.h
svm/pr_loqo.cpp.o: /usr/include/machine/machlimits.h
svm/pr_loqo.cpp.o: /usr/include/sys/syslimits.h
svm/pr_loqo.cpp.o: /usr/include/machine/machtime.h
svm/pr_loqo.cpp.o: /usr/include/sys/rt_limits.h /usr/include/stdarg.h
svm/pr_loqo.cpp.o: /usr/include/math.h /usr/include/stdlib.h
svm/pr_loqo.cpp.o: /usr/include/time.h /usr/include/sys/timers.h
svm/pr_loqo.cpp.o: /usr/include/sys/signalmisc.h /usr/include/sys/timemisc.h

hmm/HMM.c.o: lib/Mathmatics.h lib/common.h /usr/include/math.h
hmm/HMM.c.o: /usr/include/standards.h /usr/include/stdlib.h
hmm/HMM.c.o: /usr/include/getopt.h /usr/include/sys/types.h
hmm/HMM.c.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
hmm/HMM.c.o: /usr/include/sys/select.h /usr/include/stdio.h
hmm/HMM.c.o: /usr/include/sys/seek.h /usr/include/va_list.h
hmm/HMM.c.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
hmm/HMM.c.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
hmm/HMM.c.o: /usr/include/sys/rt_limits.h lib/Observation.h lib/io.h
hmm/HMM.c.o: /usr/include/stdarg.h
lib/Mathmatics.c.o: lib/common.h /usr/include/math.h /usr/include/standards.h
lib/Mathmatics.c.o: /usr/include/stdlib.h /usr/include/getopt.h
lib/Mathmatics.c.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/Mathmatics.c.o: /usr/include/mach/machine/vm_types.h
lib/Mathmatics.c.o: /usr/include/sys/select.h /usr/include/stdio.h
lib/Mathmatics.c.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Mathmatics.c.o: /usr/include/sys/limits.h
lib/Mathmatics.c.o: /usr/include/machine/machlimits.h
lib/Mathmatics.c.o: /usr/include/sys/syslimits.h
lib/Mathmatics.c.o: /usr/include/machine/machtime.h
lib/Mathmatics.c.o: /usr/include/sys/rt_limits.h
lib/Observation.c.o: /usr/include/stdio.h /usr/include/standards.h
lib/Observation.c.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/Observation.c.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/Observation.c.o: /usr/include/mach/machine/vm_types.h
lib/Observation.c.o: /usr/include/sys/select.h /usr/include/getopt.h
lib/Observation.c.o: /usr/include/sys/limits.h
lib/Observation.c.o: /usr/include/machine/machlimits.h
lib/Observation.c.o: /usr/include/sys/syslimits.h
lib/Observation.c.o: /usr/include/machine/machtime.h
lib/Observation.c.o: /usr/include/sys/rt_limits.h lib/io.h
lib/Observation.c.o: /usr/include/stdarg.h lib/common.h lib/Mathmatics.h
lib/Observation.c.o: /usr/include/math.h /usr/include/stdlib.h
lib/io.c.o: /usr/include/stdio.h /usr/include/standards.h
lib/io.c.o: /usr/include/sys/seek.h /usr/include/va_list.h
lib/io.c.o: /usr/include/sys/types.h /usr/include/time_t.h
lib/io.c.o: /usr/include/mach/machine/vm_types.h /usr/include/sys/select.h
lib/io.c.o: /usr/include/getopt.h /usr/include/sys/limits.h
lib/io.c.o: /usr/include/machine/machlimits.h /usr/include/sys/syslimits.h
lib/io.c.o: /usr/include/machine/machtime.h /usr/include/sys/rt_limits.h
lib/io.c.o: /usr/include/stdarg.h
svm/Optimizer.c.o: lib/common.h
svm/SVM.c.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.c.o: /usr/include/standards.h /usr/include/sys/seek.h
svm/SVM.c.o: /usr/include/va_list.h /usr/include/sys/types.h
svm/SVM.c.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/SVM.c.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM.c.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/SVM.c.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/SVM.c.o: /usr/include/sys/rt_limits.h lib/io.h /usr/include/stdarg.h
svm/SVM.c.o: lib/Mathmatics.h /usr/include/math.h /usr/include/stdlib.h
svm/SVM.c.o: /usr/include/ctype.h /usr/include/sys/localedef.h
svm/SVM.c.o: /usr/include/sys/lc_core.h /usr/include/locale.h
svm/SVM.c.o: /usr/include/nl_types.h /usr/include/mesg.h
svm/SVM.c.o: /usr/include/limits.h /usr/include/string.h
svm/SVM.c.o: /usr/include/strings.h /usr/include/time.h
svm/SVM.c.o: /usr/include/sys/timers.h /usr/include/sys/signalmisc.h
svm/SVM.c.o: /usr/include/sys/timemisc.h /usr/include/float.h
svm/SVM.c.o: /usr/include/fp_class.h
svm/SVM_light.c.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.c.o: /usr/include/stdio.h /usr/include/standards.h
svm/SVM_light.c.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/SVM_light.c.o: /usr/include/sys/types.h /usr/include/time_t.h
svm/SVM_light.c.o: /usr/include/mach/machine/vm_types.h
svm/SVM_light.c.o: /usr/include/sys/select.h /usr/include/getopt.h
svm/SVM_light.c.o: /usr/include/sys/limits.h
svm/SVM_light.c.o: /usr/include/machine/machlimits.h
svm/SVM_light.c.o: /usr/include/sys/syslimits.h
svm/SVM_light.c.o: /usr/include/machine/machtime.h
svm/SVM_light.c.o: /usr/include/sys/rt_limits.h lib/io.h
svm/SVM_light.c.o: /usr/include/stdarg.h lib/Mathmatics.h /usr/include/math.h
svm/SVM_light.c.o: /usr/include/stdlib.h /usr/include/ctype.h
svm/SVM_light.c.o: /usr/include/sys/localedef.h /usr/include/sys/lc_core.h
svm/SVM_light.c.o: /usr/include/locale.h /usr/include/nl_types.h
svm/SVM_light.c.o: /usr/include/mesg.h /usr/include/limits.h
svm/SVM_light.c.o: /usr/include/string.h /usr/include/strings.h
svm/SVM_light.c.o: /usr/include/time.h /usr/include/sys/timers.h
svm/SVM_light.c.o: /usr/include/sys/signalmisc.h /usr/include/sys/timemisc.h
svm/SVM_light.c.o: /usr/include/float.h /usr/include/fp_class.h svm/kernel.h
svm/SVM_light.c.o: hmm/HMM.h svm/Optimizer.h
svm/kernel.c.o: hmm/HMM.h lib/Mathmatics.h lib/common.h /usr/include/math.h
svm/kernel.c.o: /usr/include/standards.h /usr/include/stdlib.h
svm/kernel.c.o: /usr/include/getopt.h /usr/include/sys/types.h
svm/kernel.c.o: /usr/include/time_t.h /usr/include/mach/machine/vm_types.h
svm/kernel.c.o: /usr/include/sys/select.h /usr/include/stdio.h
svm/kernel.c.o: /usr/include/sys/seek.h /usr/include/va_list.h
svm/kernel.c.o: /usr/include/sys/limits.h /usr/include/machine/machlimits.h
svm/kernel.c.o: /usr/include/sys/syslimits.h /usr/include/machine/machtime.h
svm/kernel.c.o: /usr/include/sys/rt_limits.h lib/Observation.h lib/io.h
svm/kernel.c.o: /usr/include/stdarg.h
