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
