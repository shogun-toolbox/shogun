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

debug: COMPFLAGS=-g -DDEBUG
debug: LINKFLAGS=-g -DDEBUG
debug: genefinder.`uname`
	gdb genefinder.`uname`

static: LINKFLAGS+=-static
static: genefinder.`uname`

purif: COMP=$(COMP_SUN)
purif: LINK=$(LINK_SUN)
purif: COMPFLAGS=$(COMPFLAGS_SUN)
purif: LINKFLAGS=$(LINKFLAGS_SUN)
purif:	genefinder.`uname`
	/home/solaris2/purify/purify-4.2-solaris2/purify -always-use-cache-dir \
	-best-effort -cache-dir=/tmp/ccache $(COMP) */*.$(EXT_OBJ) \
	$(LINKFLAGS) -o ../genfinder.`uname`.purif

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

hmm/HMM.o: /usr/include/math.h /usr/include/features.h
hmm/HMM.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
hmm/HMM.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
hmm/HMM.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
hmm/HMM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
hmm/HMM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
hmm/HMM.o: /usr/include/bits/types.h /usr/include/libio.h
hmm/HMM.o: /usr/include/_G_config.h /usr/include/wchar.h
hmm/HMM.o: /usr/include/bits/wchar.h /usr/include/gconv.h
hmm/HMM.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
hmm/HMM.o: /usr/include/stdlib.h lib/common.h lib/Observation.h
lib/Mathmatics.o: /usr/include/math.h /usr/include/features.h
lib/Mathmatics.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/Mathmatics.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
lib/Mathmatics.o: /usr/include/bits/mathcalls.h /usr/include/stdlib.h
lib/Mathmatics.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Mathmatics.o: lib/common.h
lib/Observation.o: /usr/include/stdio.h /usr/include/features.h
lib/Observation.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/Observation.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Observation.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
lib/Observation.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Observation.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Observation.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Observation.o: /usr/include/bits/stdio_lim.h lib/common.h
lib/Observation.o: lib/Mathmatics.h /usr/include/math.h
lib/Observation.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
lib/Observation.o: /usr/include/bits/mathcalls.h /usr/include/stdlib.h
svm/Optimizer.o: lib/common.h
svm/SVM.o: lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/SVM.o: /usr/include/gnu/stubs.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h /usr/include/math.h
svm/SVM.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
svm/SVM.o: /usr/include/bits/mathcalls.h /usr/include/stdlib.h
svm/SVM.o: /usr/include/ctype.h /usr/include/endian.h
svm/SVM.o: /usr/include/bits/endian.h /usr/include/string.h
svm/SVM.o: /usr/include/time.h /usr/include/bits/time.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM_light.o: svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.o: /usr/include/stdio.h /usr/include/features.h
svm/SVM_light.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM_light.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM_light.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM_light.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM_light.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
svm/SVM_light.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM_light.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/SVM_light.o: /usr/include/stdlib.h /usr/include/ctype.h
svm/SVM_light.o: /usr/include/endian.h /usr/include/bits/endian.h
svm/SVM_light.o: /usr/include/string.h /usr/include/time.h
svm/SVM_light.o: /usr/include/bits/time.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM_light.o: svm/kernel.h hmm/HMM.h svm/Optimizer.h
svm/kernel.o: hmm/HMM.h /usr/include/math.h /usr/include/features.h
svm/kernel.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/kernel.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
svm/kernel.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
svm/kernel.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/kernel.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/kernel.o: /usr/include/bits/types.h /usr/include/libio.h
svm/kernel.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/kernel.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/kernel.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
svm/kernel.o: /usr/include/stdlib.h lib/common.h lib/Observation.h
svm_cplex/SVM_cplex.o: hmm/HMM.h /usr/include/math.h /usr/include/features.h
svm_cplex/SVM_cplex.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm_cplex/SVM_cplex.o: /usr/include/bits/huge_val.h
svm_cplex/SVM_cplex.o: /usr/include/bits/mathdef.h
svm_cplex/SVM_cplex.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
svm_cplex/SVM_cplex.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/SVM_cplex.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/SVM_cplex.o: /usr/include/bits/types.h /usr/include/libio.h
svm_cplex/SVM_cplex.o: /usr/include/_G_config.h /usr/include/wchar.h
svm_cplex/SVM_cplex.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm_cplex/SVM_cplex.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
svm_cplex/SVM_cplex.o: /usr/include/stdlib.h lib/common.h lib/Observation.h
svm_cplex/SVM_cplex.o: svm/SVM.h /usr/include/ctype.h /usr/include/endian.h
svm_cplex/SVM_cplex.o: /usr/include/bits/endian.h /usr/include/string.h
svm_cplex/SVM_cplex.o: /usr/include/time.h /usr/include/bits/time.h
svm_cplex/SVM_cplex.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/assert.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/features.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/sys/cdefs.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/gnu/stubs.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/limits.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/stdlib.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/stdio.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/types.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/libio.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/_G_config.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/wchar.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/wchar.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/gconv.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/stdio_lim.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/setjmp.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/setjmp.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/sigset.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/assert.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/features.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/sys/cdefs.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/gnu/stubs.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/limits.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/stdlib.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/stdio.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/types.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/libio.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/_G_config.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/wchar.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/wchar.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/gconv.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/stdio_lim.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/setjmp.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/setjmp.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/sigset.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/assert.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/features.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/sys/cdefs.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/gnu/stubs.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/limits.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/stdlib.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/stdio.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/types.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/libio.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/_G_config.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/wchar.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/wchar.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/gconv.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/stdio_lim.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/setjmp.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/setjmp.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/sigset.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/spdiag.o: /usr/include/assert.h /usr/include/features.h
svm_cplex/spdiag.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/spdiag.o: /usr/include/limits.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/spdiag.o: /usr/include/stdlib.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/spdiag.o: /usr/include/stdio.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/spdiag.o: /usr/include/bits/types.h /usr/include/libio.h
svm_cplex/spdiag.o: /usr/include/_G_config.h /usr/include/wchar.h
svm_cplex/spdiag.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm_cplex/spdiag.o: /usr/include/bits/stdio_lim.h /usr/include/setjmp.h
svm_cplex/spdiag.o: /usr/include/bits/setjmp.h /usr/include/bits/sigset.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/train_svm.o: /usr/include/assert.h /usr/include/features.h
svm_cplex/train_svm.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/train_svm.o: /usr/include/limits.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/train_svm.o: /usr/include/stdlib.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/train_svm.o: /usr/include/stdio.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/train_svm.o: /usr/include/bits/types.h /usr/include/libio.h
svm_cplex/train_svm.o: /usr/include/_G_config.h /usr/include/wchar.h
svm_cplex/train_svm.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm_cplex/train_svm.o: /usr/include/bits/stdio_lim.h /usr/include/setjmp.h
svm_cplex/train_svm.o: /usr/include/bits/setjmp.h /usr/include/bits/sigset.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/libsgl.h
gui/GeneFinder.o: /usr/include/stdio.h /usr/include/features.h
gui/GeneFinder.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
gui/GeneFinder.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
gui/GeneFinder.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
gui/GeneFinder.o: /usr/include/bits/types.h /usr/include/libio.h
gui/GeneFinder.o: /usr/include/_G_config.h /usr/include/wchar.h
gui/GeneFinder.o: /usr/include/bits/wchar.h /usr/include/gconv.h
gui/GeneFinder.o: /usr/include/bits/stdio_lim.h /usr/include/unistd.h
gui/GeneFinder.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
gui/GeneFinder.o: /usr/include/time.h /usr/include/bits/time.h
gui/GeneFinder.o: /usr/include/string.h /usr/include/ctype.h
gui/GeneFinder.o: /usr/include/endian.h /usr/include/bits/endian.h
gui/GeneFinder.o: /usr/include/stdlib.h /usr/include/math.h
gui/GeneFinder.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
gui/GeneFinder.o: /usr/include/bits/mathcalls.h /usr/include/fcntl.h
gui/GeneFinder.o: /usr/include/bits/fcntl.h /usr/include/sys/types.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/libmmfile.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/libmatlb.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/matrix.h
gui/GeneFinder.o: /usr/include/assert.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/tmwtypes.h
gui/GeneFinder.o: /usr/include/limits.h
gui/GeneFinder.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
gui/GeneFinder.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/mwutil.h
gui/GeneFinder.o: /usr/include/setjmp.h /usr/include/bits/setjmp.h
gui/GeneFinder.o: /usr/include/bits/sigset.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/mex.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/libmatlbm.h
gui/GeneFinder.o: /home/matrix/matlab6/extern/include/libsgl.h hmm/HMM.h
gui/GeneFinder.o: lib/Mathmatics.h lib/common.h lib/Observation.h svm/SVM.h
gui/GeneFinder.o: svm/SVM_light.h svm/kernel.h svm/Optimizer.h
gui/GeneFinder.o: svm_cplex/SVM_cplex.h
hmm/HMM.o: hmm/HMM.h /usr/include/math.h /usr/include/features.h
hmm/HMM.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
hmm/HMM.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
hmm/HMM.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
hmm/HMM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
hmm/HMM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
hmm/HMM.o: /usr/include/bits/types.h /usr/include/libio.h
hmm/HMM.o: /usr/include/_G_config.h /usr/include/wchar.h
hmm/HMM.o: /usr/include/bits/wchar.h /usr/include/gconv.h
hmm/HMM.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
hmm/HMM.o: /usr/include/stdlib.h lib/common.h lib/Observation.h
hmm/HMM.o: /usr/include/time.h /usr/include/bits/time.h /usr/include/ctype.h
hmm/HMM.o: /usr/include/endian.h /usr/include/bits/endian.h
lib/Mathmatics.o: /usr/include/stdio.h /usr/include/features.h
lib/Mathmatics.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/Mathmatics.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Mathmatics.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
lib/Mathmatics.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Mathmatics.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Mathmatics.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Mathmatics.o: /usr/include/bits/stdio_lim.h /usr/include/math.h
lib/Mathmatics.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
lib/Mathmatics.o: /usr/include/bits/mathcalls.h /usr/include/time.h
lib/Mathmatics.o: /usr/include/bits/time.h /usr/include/stdlib.h
lib/Mathmatics.o: lib/Mathmatics.h lib/common.h
lib/Observation.o: /usr/include/stdio.h /usr/include/features.h
lib/Observation.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
lib/Observation.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
lib/Observation.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
lib/Observation.o: /usr/include/bits/types.h /usr/include/libio.h
lib/Observation.o: /usr/include/_G_config.h /usr/include/wchar.h
lib/Observation.o: /usr/include/bits/wchar.h /usr/include/gconv.h
lib/Observation.o: /usr/include/bits/stdio_lim.h lib/Observation.h
lib/Observation.o: lib/common.h lib/Mathmatics.h /usr/include/math.h
lib/Observation.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
lib/Observation.o: /usr/include/bits/mathcalls.h /usr/include/stdlib.h
svm/Optimizer.o: svm/pr_loqo.h svm/Optimizer.h lib/common.h svm/SVM.h
svm/Optimizer.o: lib/Observation.h /usr/include/stdio.h
svm/Optimizer.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/Optimizer.o: /usr/include/gnu/stubs.h
svm/Optimizer.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/Optimizer.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/Optimizer.o: /usr/include/bits/types.h /usr/include/libio.h
svm/Optimizer.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/Optimizer.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/Optimizer.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
svm/Optimizer.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/Optimizer.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/Optimizer.o: /usr/include/stdlib.h /usr/include/ctype.h
svm/Optimizer.o: /usr/include/endian.h /usr/include/bits/endian.h
svm/Optimizer.o: /usr/include/string.h /usr/include/time.h
svm/Optimizer.o: /usr/include/bits/time.h
svm/Optimizer.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM.o: svm/SVM.h lib/common.h lib/Observation.h /usr/include/stdio.h
svm/SVM.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/SVM.o: /usr/include/gnu/stubs.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h /usr/include/math.h
svm/SVM.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
svm/SVM.o: /usr/include/bits/mathcalls.h /usr/include/stdlib.h
svm/SVM.o: /usr/include/ctype.h /usr/include/endian.h
svm/SVM.o: /usr/include/bits/endian.h /usr/include/string.h
svm/SVM.o: /usr/include/time.h /usr/include/bits/time.h
svm/SVM.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM_light.o: svm/SVM_light.h svm/SVM.h lib/common.h lib/Observation.h
svm/SVM_light.o: /usr/include/stdio.h /usr/include/features.h
svm/SVM_light.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/SVM_light.o: /usr/include/bits/types.h /usr/include/libio.h
svm/SVM_light.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/SVM_light.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/SVM_light.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
svm/SVM_light.o: /usr/include/math.h /usr/include/bits/huge_val.h
svm/SVM_light.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/SVM_light.o: /usr/include/stdlib.h /usr/include/ctype.h
svm/SVM_light.o: /usr/include/endian.h /usr/include/bits/endian.h
svm/SVM_light.o: /usr/include/string.h /usr/include/time.h
svm/SVM_light.o: /usr/include/bits/time.h
svm/SVM_light.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm/SVM_light.o: svm/kernel.h hmm/HMM.h svm/Optimizer.h
svm/kernel.o: svm/kernel.h hmm/HMM.h /usr/include/math.h
svm/kernel.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm/kernel.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
svm/kernel.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
svm/kernel.o: /usr/include/stdio.h
svm/kernel.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/kernel.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/kernel.o: /usr/include/bits/types.h /usr/include/libio.h
svm/kernel.o: /usr/include/_G_config.h /usr/include/wchar.h
svm/kernel.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm/kernel.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
svm/kernel.o: /usr/include/stdlib.h lib/common.h lib/Observation.h
svm/pr_loqo.o: /usr/include/math.h /usr/include/features.h
svm/pr_loqo.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm/pr_loqo.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
svm/pr_loqo.o: /usr/include/bits/mathcalls.h /usr/include/time.h
svm/pr_loqo.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm/pr_loqo.o: /usr/include/bits/time.h /usr/include/bits/types.h
svm/pr_loqo.o: /usr/include/stdlib.h /usr/include/stdio.h
svm/pr_loqo.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm/pr_loqo.o: /usr/include/libio.h /usr/include/_G_config.h
svm/pr_loqo.o: /usr/include/wchar.h /usr/include/bits/wchar.h
svm/pr_loqo.o: /usr/include/gconv.h /usr/include/bits/stdio_lim.h
svm/pr_loqo.o: svm/pr_loqo.h
svm_cplex/SVM_cplex.o: svm_cplex/SVM_cplex.h hmm/HMM.h /usr/include/math.h
svm_cplex/SVM_cplex.o: /usr/include/features.h /usr/include/sys/cdefs.h
svm_cplex/SVM_cplex.o: /usr/include/gnu/stubs.h /usr/include/bits/huge_val.h
svm_cplex/SVM_cplex.o: /usr/include/bits/mathdef.h
svm_cplex/SVM_cplex.o: /usr/include/bits/mathcalls.h /usr/include/stdio.h
svm_cplex/SVM_cplex.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/SVM_cplex.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/SVM_cplex.o: /usr/include/bits/types.h /usr/include/libio.h
svm_cplex/SVM_cplex.o: /usr/include/_G_config.h /usr/include/wchar.h
svm_cplex/SVM_cplex.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm_cplex/SVM_cplex.o: /usr/include/bits/stdio_lim.h lib/Mathmatics.h
svm_cplex/SVM_cplex.o: /usr/include/stdlib.h lib/common.h lib/Observation.h
svm_cplex/SVM_cplex.o: svm/SVM.h /usr/include/ctype.h /usr/include/endian.h
svm_cplex/SVM_cplex.o: /usr/include/bits/endian.h /usr/include/string.h
svm_cplex/SVM_cplex.o: /usr/include/time.h /usr/include/bits/time.h
svm_cplex/SVM_cplex.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/SVM_cplex.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/SVM_cplex.o: /usr/include/assert.h
svm_cplex/SVM_cplex.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/SVM_cplex.o: /usr/include/limits.h
svm_cplex/SVM_cplex.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/SVM_cplex.o: svm_cplex/train_svm.h
svm_cplex/SVM_cplex.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/SVM_cplex.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/SVM_cplex.o: /usr/include/setjmp.h /usr/include/bits/setjmp.h
svm_cplex/SVM_cplex.o: /usr/include/bits/sigset.h
svm_cplex/SVM_cplex.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/SVM_cplex.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/SVM_cplex.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/cplex_init_mex_interface.o: svm_cplex/cplex_init_mex_interface.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/assert.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/features.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/sys/cdefs.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/gnu/stubs.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/limits.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/stdlib.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/stdio.h
svm_cplex/cplex_init_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/types.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/libio.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/_G_config.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/wchar.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/wchar.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/gconv.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/stdio_lim.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/setjmp.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/setjmp.h
svm_cplex/cplex_init_mex_interface.o: /usr/include/bits/sigset.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/cplex_init_mex_interface.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/cplex_quit_mex_interface.o: svm_cplex/cplex_quit_mex_interface.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/assert.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/features.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/sys/cdefs.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/gnu/stubs.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/limits.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/stdlib.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/stdio.h
svm_cplex/cplex_quit_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/types.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/libio.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/_G_config.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/wchar.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/wchar.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/gconv.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/stdio_lim.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/setjmp.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/setjmp.h
svm_cplex/cplex_quit_mex_interface.o: /usr/include/bits/sigset.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/cplex_quit_mex_interface.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/qp_solve_mex_interface.o: svm_cplex/qp_solve_mex_interface.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/assert.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/features.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/sys/cdefs.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/gnu/stubs.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/limits.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/stdlib.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/stdio.h
svm_cplex/qp_solve_mex_interface.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/types.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/libio.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/_G_config.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/wchar.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/wchar.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/gconv.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/stdio_lim.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/setjmp.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/setjmp.h
svm_cplex/qp_solve_mex_interface.o: /usr/include/bits/sigset.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/qp_solve_mex_interface.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/spdiag.o: svm_cplex/spdiag.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/spdiag.o: /usr/include/assert.h /usr/include/features.h
svm_cplex/spdiag.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/spdiag.o: /usr/include/limits.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/spdiag.o: /usr/include/stdlib.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/spdiag.o: /usr/include/stdio.h
svm_cplex/spdiag.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/spdiag.o: /usr/include/bits/types.h /usr/include/libio.h
svm_cplex/spdiag.o: /usr/include/_G_config.h /usr/include/wchar.h
svm_cplex/spdiag.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm_cplex/spdiag.o: /usr/include/bits/stdio_lim.h /usr/include/setjmp.h
svm_cplex/spdiag.o: /usr/include/bits/setjmp.h /usr/include/bits/sigset.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/spdiag.o: /home/matrix/matlab6/extern/include/libmmfile.h
svm_cplex/train_svm.o: svm_cplex/train_svm.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/train_svm.o: /usr/include/assert.h /usr/include/features.h
svm_cplex/train_svm.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/train_svm.o: /usr/include/limits.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/train_svm.o: /usr/include/stdlib.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/train_svm.o: /usr/include/stdio.h
svm_cplex/train_svm.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/train_svm.o: /usr/include/bits/types.h /usr/include/libio.h
svm_cplex/train_svm.o: /usr/include/_G_config.h /usr/include/wchar.h
svm_cplex/train_svm.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm_cplex/train_svm.o: /usr/include/bits/stdio_lim.h /usr/include/setjmp.h
svm_cplex/train_svm.o: /usr/include/bits/setjmp.h /usr/include/bits/sigset.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/train_svm.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/train_svm.o: svm_cplex/cplex_init_mex_interface.h
svm_cplex/train_svm.o: svm_cplex/qp_solve_mex_interface.h svm_cplex/spdiag.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/libmatlb.h
svm_cplex/train_svm_main.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stddef.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/matrix.h
svm_cplex/train_svm_main.o: /usr/include/assert.h /usr/include/features.h
svm_cplex/train_svm_main.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/tmwtypes.h
svm_cplex/train_svm_main.o: /usr/include/limits.h
svm_cplex/train_svm_main.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/limits.h
svm_cplex/train_svm_main.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/float.h
svm_cplex/train_svm_main.o: /usr/include/stdlib.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/mwutil.h
svm_cplex/train_svm_main.o: /usr/include/stdio.h
svm_cplex/train_svm_main.o: /usr/lib/gcc-lib/i486-suse-linux/2.95.2/include/stdarg.h
svm_cplex/train_svm_main.o: /usr/include/bits/types.h /usr/include/libio.h
svm_cplex/train_svm_main.o: /usr/include/_G_config.h /usr/include/wchar.h
svm_cplex/train_svm_main.o: /usr/include/bits/wchar.h /usr/include/gconv.h
svm_cplex/train_svm_main.o: /usr/include/bits/stdio_lim.h
svm_cplex/train_svm_main.o: /usr/include/setjmp.h /usr/include/bits/setjmp.h
svm_cplex/train_svm_main.o: /usr/include/bits/sigset.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/mex.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/libmatlbm.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/libsgl.h
svm_cplex/train_svm_main.o: svm_cplex/train_svm.h
svm_cplex/train_svm_main.o: svm_cplex/cplex_init_mex_interface.h
svm_cplex/train_svm_main.o: svm_cplex/qp_solve_mex_interface.h
svm_cplex/train_svm_main.o: svm_cplex/spdiag.h
svm_cplex/train_svm_main.o: /home/matrix/matlab6/extern/include/libmmfile.h
