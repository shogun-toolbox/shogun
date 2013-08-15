.PHONY: doc clean distclean
EXAMPLE_FILES = $(shell find ../examples -name '*.py' -o -name '*.m' -o -name\
				'*.R' -o -name '*.cs' -o -name '*.java' -o -name '*.lua' -o \
				-name '*.rb' -type f)
UNDOCUMENTED_EXAMPLE_DIRS = $(shell find ../examples/undocumented -type d | \
							grep -v svn)
DOCUMENTED_EXAMPLE_DIRS = $(patsubst ../examples/undocumented%,../examples/documented%,\
						  $(UNDOCUMENTED_EXAMPLE_DIRS))
MAINVERSION := $(shell awk '/Release/{print $$5;exit}' ../NEWS)


doc: examples pages/Configure.generated Doxyfile_en Doxyfile_cn
	doxygen Doxyfile_en
	doxygen Doxyfile_cn

Doxyfile_en: Doxyfile_en.in
	sed "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = v$(MAINVERSION)/" Doxyfile_en.in >Doxyfile_en

Doxyfile_cn: Doxyfile_cn.in
	sed "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = v$(MAINVERSION)/" Doxyfile_cn.in >Doxyfile_cn

pages/Configure.generated:	
	cmake --help 2>&1 >$@

doc_cn: examples_cn Doxyfile_cn
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

put-online: doc
	test -n "$(MAINVERSION)"
	ssh km rm -rf "/var/www/shogun-toolbox.org/doc/en/$(MAINVERSION)"
	ssh km mkdir "/var/www/shogun-toolbox.org/doc/en/$(MAINVERSION)"
	cd html && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh km tar -C /var/www/shogun-toolbox.org/doc/en/$(MAINVERSION) -xjvf -
	ssh km find /var/www/shogun-toolbox.org/doc/en/$(MAINVERSION) -type f -exec chmod 644 \{\} "\;"
	ssh km find /var/www/shogun-toolbox.org/doc/en/$(MAINVERSION) -type d -exec chmod 755 \{\} "\;"
	#
	ssh km rm -rf "/var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION)"
	ssh km mkdir "/var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION)"
	cd html_cn && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh km tar -C /var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION) -xjvf -
	ssh km find /var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION) -type f -exec chmod 644 \{\} "\;"
	ssh km find /var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION) -type d -exec chmod 755 \{\} "\;"
