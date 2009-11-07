.PHONY: doc clean distclean
EXAMPLE_FILES = $(shell find ../examples -name '*.py' -o -name '*.m' -o -name '*.R' -type f | grep -v svn)

doc: examples
	doxygen Doxyfile

doc_cn: examples_cn
	doxygen Doxyfile_cn

examples: $(EXAMPLE_FILES)
	python ./insert_examples.py pages/

examples_cn: $(EXAMPLE_FILES) pages_cn/
	python ./insert_examples.py cn

distclean: clean

clean: 
	rm -rf html
