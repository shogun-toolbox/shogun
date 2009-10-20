.PHONY: doc clean distclean
EXAMPLE_FILES = $(shell find ../examples -name '*.py' -o -name '*.m' -o -name '*.R' -type f | grep -v svn)

doc: examples
	doxygen

examples: $(EXAMPLE_FILES)
	python ./insert_examples.py

distclean: clean

clean: 
	rm -rf html
