.PHONY: doc clean distclean
EXAMPLE_FILES = $(shell find ../examples -name '*.py' -o -name '*.m' -o -name\
				'*.R' -o -name '*.cs' -o -name '*.java' -o -name '*.lua' -o \
				-name '*.rb' -type f)
UNDOCUMENTED_EXAMPLE_DIRS = $(shell find ../examples/undocumented -type d | \
							grep -v svn)
DOCUMENTED_EXAMPLE_DIRS = $(patsubst ../examples/undocumented%,../examples/documented%,\
						  $(UNDOCUMENTED_EXAMPLE_DIRS))

doc: examples pages/Configure.generated
	doxygen Doxyfile

pages/Configure.generated:	../src/configure
	../src/configure --help 2>&1 >$@

doc_cn: examples_cn
	doxygen Doxyfile_cn

examples: $(UNDOCUMENTED_EXAMPLE_DIRS) $(EXAMPLE_FILES)
	$(MAKE) -C ../examples
	echo $(DOCUMENTED_EXAMPLE_DIRS)
	python ./insert_examples.py pages/
	python ./insert_examples_cn.py pages_cn/

distclean: clean

clean: 
	rm -rf html html_cn
	rm -f pages/Configure.generated
