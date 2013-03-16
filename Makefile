# Usage scenarios
#
# * To make a release (and tag it) run
#
#       make prepare-release
#       make git-tag-release  
#       make release
#       make data-release
#       make update-webpage
#
# * To create a debian .orig.tar.gz run
#
#       make release DEBIAN=yes
#
# * To create a snapshot run
#
#       make release SNAPSHOT=yes
# 
# * To sign/md5sum the created tarballs and copy them to the webpage
#
#       make update-webpage
#
# * To make a release from inside a tarball
#
#       make package-from-release
#
#
# The following additional options may be set
#
# DEBIAN=yes   						-> use debian naming scheme shogun_0.1.1+git2f0a2c8.orig.tar.gz
# SVMLIGHT=no						-> remove svm light
# COMPRESS=cruncher					-> use cruncher (bz2/gz etc as file compressor)
# SNAPSHOT=yes						-> use git snapshot naming scheme
# MAINVERSION=0.2.0					-> main version
# EXTRAVERSION=+git2f0a2c8			-> extra version string
# RELEASENAME=shogun-3.0.0+extra	-> use different releasename, here shogun-3.0.0+extra
#
# * For example to use gzip instead of bzip2 and to append an extra version
# string.
#       make release COMPRESS=gzip EXTRAVERSION=+git2f0a2c8
#
# * To create a debian snapshot package
#
#      make DEBIAN=yes SNAPSHOT=yes
#

DEBIAN := no
SVMLIGHT := yes
COMPRESS := bzip2
MAINVERSION := $(shell awk '/Release/{print $$5;exit}' src/NEWS)
VERSIONBASE := $(shell echo $(MAINVERSION) | cut -f 1-2 -d '.')
EXTRAVERSION := 
DATAMAINVERSION := $(shell awk '/Release/{print $$9;exit}' src/NEWS | tr -d '(,)' )
DATAEXTRAVERSION :=
DATARELEASENAME := shogun-data-$(DATAMAINVERSION)$(DATAEXTRAVERSION)
RELEASENAME := shogun-$(MAINVERSION)$(EXTRAVERSION)
GITVERSION := $(shell git show --pretty='format:%h'|head -1)
LOGFILE := 'prepare-release.log'

stop:
	@echo
	@echo "To install shogun, please go to the 'src' directory,"
	@echo "most of the time"
	@echo
	@echo "cd src"
	@echo "./configure"
	@echo "make"
	@echo
	@echo "which will build all available interfaces of shogun"
	@echo "should be enough. For further information consult"
	@echo "especially the INSTALL and README files and try"
	@echo
	@echo "./configure --help"
	@echo

debian-package:	DEBIAN=yes
ifeq ($(DEBIAN),yes)
COMPRESS := gzip
SVMLIGHT := no
RELEASENAME := shogun_$(MAINVERSION)
ifeq ($(SNAPSHOT),yes)
RELEASENAME := $(RELEASENAME)+git$(GITVERSION)
endif
RELEASENAME := $(RELEASENAME).orig
else
ifeq ($(SNAPSHOT),yes)
RELEASENAME := $(RELEASENAME)+git$(GITVERSION)
endif
all: doc release matlab python octave
endif


.PHONY: all release package-from-release update-webpage clean distclean embed-main-version

DATADESTDIR := ../shogun-releases/$(DATARELEASENAME)
DESTDIR := ../shogun-releases/$(RELEASENAME)
REMOVE_SVMLIGHT := rm -f $(DESTDIR)/src/shogun/classifier/svm/SVM_light.* $(DESTDIR)/src/shogun/classifier/svm/Optimizer.* $(DESTDIR)/src/shogun/regression/svr/SVR_light.* $(DESTDIR)/src/LICENSE.SVMlight; \
rm -f $(DESTDIR)/testsuite/data/classifier/SVMLight* $(DESTDIR)/testsuite/data/regression/SVRLight*	; \
grep -rl USE_SVMLIGHT $(DESTDIR)| xargs --no-run-if-empty sed -i '/\#ifdef USE_SVMLIGHT/,/\#endif \/\/USE_SVMLIGHT/c \\' ; \
sed -i '/^ \* EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS/,/ \* this program is free software/c\ * This program is free software; you can redistribute it and/or modify' $(DESTDIR)/src/shogun/kernel/Kernel.cpp $(DESTDIR)/src/shogun/kernel/Kernel.h ; \
sed -i '/^SVMlight:$$/,/^$$/c\\' $(DESTDIR)/src/LICENSE

# We assume that a release is always created from a git clone.

prepare-release:
	@if [ -f $(LOGFILE) ]; then rm -f $(LOGFILE); fi
	git pull --rebase github master | tee -a $(LOGFILE)
	git submodule update | tee -a $(LOGFILE)
	#update changelog
	git status -v | tee -a $(LOGFILE)
	@echo | tee -a $(LOGFILE)
	@echo "Please check output of 'git status'." | tee -a $(LOGFILE)
	@echo "Press ENTER to continue or Ctrl-C to abort." | tee -a $(LOGFILE)
	@read foobar
	(cd src &&  rm -f ChangeLog ; ./configure ; $(MAKE) ChangeLog && git commit -m "updated changelog" ChangeLog ) | tee -a $(LOGFILE)
	#build for all interfaces and update doc
	git clean -dfx
	( cd src && ./configure ) | tee -a $(LOGFILE)
	+$(MAKE) -C src | tee -a $(LOGFILE)
	+$(MAKE) -C src install DESTDIR=/tmp/sg_test_build | tee -a $(LOGFILE)
	+$(MAKE) -C src doc DESTDIR=/tmp/sg_test_build | tee -a $(LOGFILE)
	-$(MAKE) -C src tests DESTDIR=/tmp/sg_test_build | tee -a $(LOGFILE)
	#$(MAKE) -C src distclean | tee --append $(LOGFILE)
	git commit doc -m "updated reference documentation" | tee -a $(LOGFILE)

release: src/shogun/lib/versionstring.h $(DESTDIR)/src/shogun/lib/versionstring.h
	rm -rf $(DESTDIR)
	# copy things over to shogun-releases dir
	git checkout-index --prefix=$(DESTDIR)/ -a
	cp src/shogun/lib/versionstring.h $(DESTDIR)/src/shogun/lib/versionstring.h
	tar -c -f $(DESTDIR).tar -C ../shogun-releases $(RELEASENAME)
	rm -f $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	$(COMPRESS) -9 $(DESTDIR).tar

data-release:
	cd data && git checkout-index --prefix=../$(DATADESTDIR)/ -a
	rm -f $(DATADESTDIR).tar.bz2 $(DATADESTDIR).tar
	tar -c -f $(DATARELEASENAME).tar -C ../shogun-releases $(DATARELEASENAME)
	$(COMPRESS) -9 $(DATADESTDIR).tar

embed-main-version: src/shogun/lib/versionstring.h
	sed -i 's/VERSION_RELEASE "git/VERSION_RELEASE "v$(MAINVERSION)/' src/shogun/lib/versionstring.h
	sed -i "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = v$(MAINVERSION)/" doc/Doxyfile

git-tag-release: embed-main-version
	git commit -a -m "Preparing for new Release shogun_$(MAINVERSION)"
	# create shogun X.Y branch and put in versionstring
	git checkout -b shogun_$(VERSIONBASE)
	git add -f src/shogun/lib/versionstring.h
	sed -i "s| lib/versionstring.h||" src/Makefile
	git commit -a -m "Tagging shogun_$(MAINVERSION) release"
	git tag shogun_$(MAINVERSION)

package-from-release:
	rm -rf $(DESTDIR)
	mkdir $(DESTDIR)
	find ./ -regextype posix-egrep  ! -regex '(.*svn.*|.*\.o$$|.*wrap.*|.*\.so$$|.*\.mat$$|.*\.pyc$$|.*\.log$$)' \
		| xargs tar --no-recursion -cf - | tar -C $(DESTDIR) -xf - 
	if test ! $(SVMLIGHT) = yes; then $(REMOVE_SVMLIGHT); fi
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	rm -f $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	$(COMPRESS) -9 $(DESTDIR).tar

update-webpage: 
	md5sum $(DESTDIR).tar.bz2 >$(DESTDIR).md5sum
	md5sum $(DATADESTDIR).tar.bz2 >$(DATADESTDIR).md5sum
	gpg --no-emit-version -s -b -a $(DESTDIR).tar.bz2
	gpg --no-emit-version -s -b -a $(DATADESTDIR).tar.bz2

	ssh 7nn.de mkdir -m 0755 -p /var/www/shogun-toolbox.org/archives/shogun/releases/$(VERSIONBASE)/sources
	rsync --progress $(DATADESTDIR).tar.bz2 $(DATADESTDIR).md5sum \
		7nn.de:/var/www/shogun-toolbox.org/archives/shogun/data/
	rsync --progress $(DESTDIR).tar.bz2 $(DESTDIR).tar.bz2.asc $(DESTDIR).md5sum \
		7nn.de:/var/www/shogun-toolbox.org/archives/shogun/releases/$(VERSIONBASE)/sources/
	ssh 7nn.de chmod 644 "/var/www/shogun-toolbox.org/archives/shogun/releases/$(VERSIONBASE)/sources/*.* /var/www/shogun-toolbox.org/archives/shogun/data/*"
	
	$(MAKE) -C examples
	rm -rf doc/html
	$(MAKE) -C doc
	ssh 7nn.de rm -f "/var/www/shogun-toolbox.org/doc/*/$(MAINVERSION)/*.*"
	ssh 7nn.de mkdir -p "/var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION)"
	ssh 7nn.de mkdir -p "/var/www/shogun-toolbox.org/doc/en/$(MAINVERSION)"
	cd doc/html && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh 7nn.de tar -C /var/www/shogun-toolbox.org/doc/en/$(MAINVERSION)/ -xjvf -
	cd doc/html_cn && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh 7nn.de tar -C /var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION)/ -xjvf -
	ssh 7nn.de find /var/www/shogun-toolbox.org/doc/en/$(MAINVERSION) -type f -exec chmod 644 \{\} "\;"
	ssh 7nn.de find /var/www/shogun-toolbox.org/doc/en/$(MAINVERSION) -type d -exec chmod 755 \{\} "\;"
	ssh 7nn.de find /var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION) -type f -exec chmod 644 \{\} "\;"
	ssh 7nn.de find /var/www/shogun-toolbox.org/doc/cn/$(MAINVERSION) -type d -exec chmod 755 \{\} "\;"
	ssh 7nn.de ./bin/shogun_doc_install.sh $(MAINVERSION)
	rm -rf doc/html*

	cd ../shogun-publicity/website && $(MAKE)

src/shogun/lib/versionstring.h:
	rm -f src/ChangeLog
	$(MAKE) -C src ChangeLog
	$(MAKE) -C src/shogun lib/versionstring.h

$(DESTDIR)/src/shogun/lib/versionstring.h: src/shogun/lib/versionstring.h
	rm -rf $(DESTDIR)
	git checkout-index --prefix=$(DESTDIR)/ -a
	if test ! $(SVMLIGHT) = yes; then $(REMOVE_SVMLIGHT); fi

	# remove top level makefile from distribution
	cp -f src/shogun/lib/versionstring.h $(DESTDIR)/src/shogun/lib/

clean:
	rm -rf $(DESTDIR)

distclean:
	$(MAKE) -C src distclean
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	rm -rf $(DATADESTDIR) $(DATADESTDIR).tar.bz2 $(DATADESTDIR).tar.gz
