# Usage scenarios
#
# * To make a release (and tag it) run
#
#       make prepare-release
#       make svn-tag-release  
#       (cd releases/shogun_X.Y.Z ; make release ; make update-webpage)
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
#
# The following additional options may be set
#
# DEBIAN=yes   						-> use debian naming scheme shogun_0.1.1+svn1337.orig.tar.gz
# SVMLIGHT=no						-> remove svm light
# COMPRESS=cruncher					-> use cruncher (bz2/gz etc as file compressor)
# SNAPSHOT=yes						-> use svn snapshot naming scheme
# MAINVERSION=0.2.0					-> main version
# EXTRAVERSION=+svn20061202			-> extra version string
# RELEASENAME=shogun-3.0.0+extra	-> use different releasename, here shogun-3.0.0+extra
#
# * For example to use gzip instead of bzip2 and to append an extra version
# string.
#       make release COMPRESS=gzip EXTRAVERSION=+svn20061202
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
RELEASENAME := shogun-$(MAINVERSION)$(EXTRAVERSION)
SVNVERSION = $(shell svn info | grep Revision: | cut -d ' ' -f 2)

stop:
	@echo
	@echo "To install shogun, please go to the 'src' directory,"
	@echo "most of the time"
	@echo
	@echo "cd src"
	@echo "./configure --interface=<INTERFACE>"
	@echo "make"
	@echo
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
RELEASENAME := $(RELEASENAME)+svn$(SVNVERSION)
endif
RELEASENAME := $(RELEASENAME).orig
else
ifeq ($(SNAPSHOT),yes)
RELEASENAME := $(RELEASENAME)+svn$(SVNVERSION)
endif
all: doc release matlab python octave
endif


.PHONY: all release package-from-release update-webpage svn-ignores clean distclean

DESTDIR := ../$(RELEASENAME)
REMOVE_SVMLIGHT := rm -f $(DESTDIR)/src/classifier/svm/SVM_light.* $(DESTDIR)/src/classifier/svm/Optimizer.* $(DESTDIR)/src/regression/svr/SVR_light.* $(DESTDIR)/src/LICENSE.SVMlight; \
rm -f $(DESTDIR)/testsuite/data/classifier/SVMLight* $(DESTDIR)/testsuite/data/regression/SVRLight*	; \
grep -rl USE_SVMLIGHT $(DESTDIR)| xargs --no-run-if-empty sed -i '/\#ifdef USE_SVMLIGHT/,/\#endif \/\/USE_SVMLIGHT/c \\' ; \
sed -i '/^ \* EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS/,/ \* this program is free software/c\ * This program is free software; you can redistribute it and/or modify' $(DESTDIR)/src/kernel/Kernel.cpp $(DESTDIR)/src/kernel/Kernel.h ; \
sed -i '/^SVMlight:$$/,/^$$/c\\' $(DESTDIR)/src/LICENSE

# We assume that a release is always created from a SVN working copy.

prepare-release:
	svn update
	#update changelog
	+(cd src;  rm -f ChangeLog ; $(MAKE) ChangeLog ; svn ci -m "updated changelog")
	#static interfaces
	+$(MAKE) -C src distclean
	( cd src && ./configure --interface=cmdline )
	+$(MAKE) -C src 
	+$(MAKE) -C src install DESTDIR=/tmp/
	+$(MAKE) -C src reference DESTDIR=/tmp/
	+$(MAKE) -C src tests DESTDIR=/tmp/
	+$(MAKE) -C src distclean
	( cd src && ./configure --interface=octave )
	+$(MAKE) -C src 
	+$(MAKE) -C src install DESTDIR=/tmp/
	+$(MAKE) -C src reference DESTDIR=/tmp/
	+$(MAKE) -C src tests DESTDIR=/tmp/
	+$(MAKE) -C src distclean
	( cd src && ./configure --interface=python )
	+$(MAKE) -C src 
	+$(MAKE) -C src install DESTDIR=/tmp/
	+$(MAKE) -C src reference DESTDIR=/tmp/
	+$(MAKE) -C src tests DESTDIR=/tmp/
	+$(MAKE) -C src distclean
	( cd src && ./configure --interface=r )
	+$(MAKE) -C src 
	+$(MAKE) -C src install DESTDIR=/tmp/
	+$(MAKE) -C src reference DESTDIR=/tmp/
	+$(MAKE) -C src tests DESTDIR=/tmp/
	#modular interfaces
	+$(MAKE) -C src distclean
	( cd src && ./configure --interface=octave-modular )
	+$(MAKE) -C src 
	+$(MAKE) -C src install DESTDIR=/tmp/
	+$(MAKE) -C src tests DESTDIR=/tmp/
	+$(MAKE) -C src distclean
	( cd src && ./configure --interface=python-modular )
	+$(MAKE) -C src 
	+$(MAKE) -C src install DESTDIR=/tmp/
	+$(MAKE) -C src tests DESTDIR=/tmp/
	+$(MAKE) -C src distclean
	( cd src && ./configure --interface=r-modular )
	+$(MAKE) -C src 
	+$(MAKE) -C src install DESTDIR=/tmp/
	+$(MAKE) -C src tests DESTDIR=/tmp/
	(cd doc; svn ci -m "updated reference documentation")

release: src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	rm -f $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	$(COMPRESS) -9 $(DESTDIR).tar

svn-tag-release: src/lib/versionstring.h
	sed -i 's/VERSION_RELEASE "svn/VERSION_RELEASE "v$(MAINVERSION)/' src/lib/versionstring.h
	sed -i "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = v$(MAINVERSION)/" doc/Doxyfile
	svn ci -m "Preparing for new Release shogun_$(MAINVERSION)"
	-cd .. && svn --force rm releases/shogun_$(MAINVERSION)
	-cd .. && svn commit releases -m "clean old tag"
	#cd .. && svn cp trunk releases/shogun_$(MAINVERSION)
	#cp src/lib/versionstring.h ../releases/shogun_$(MAINVERSION)/src/lib/versionstring.h
	#sed -i "s| lib/versionstring.h||" ../releases/shogun_$(MAINVERSION)/src/Makefile
	#cd ../releases && svn add shogun_$(MAINVERSION)
	#cd ../releases && svn ci -m "Tagging shogun_$(MAINVERSION) release"

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
	gpg --sign $(DESTDIR).tar.bz2

	ssh vserver mkdir -m 0755 -p /pub/shogun-ftp/releases/$(VERSIONBASE)/sources
	scp $(DESTDIR).tar.bz2 $(DESTDIR).tar.bz2.gpg $(DESTDIR).md5sum \
		vserver:/pub/shogun-ftp/releases/$(VERSIONBASE)/sources/
	ssh vserver chmod 644 /pub/shogun-ftp/releases/$(VERSIONBASE)/sources/*.*
	
	rm -rf doc/html
	make -C doc
	ssh vserver rm -f /pub/shogun/doc/*.*
	cd doc/html && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh vserver tar -C /pub/shogun/doc/ -xjvf -
	ssh vserver chmod 644 /pub/shogun/doc/*.*
	ssh vserver ./bin/shogun_doc_install.sh
	rm -rf doc/html

src/lib/versionstring.h:
	rm -f src/ChangeLog
	make -C src ChangeLog
	make -C src lib/versionstring.h

$(DESTDIR)/src/lib/versionstring.h: src/lib/versionstring.h
	rm -rf $(DESTDIR)
	svn export . $(DESTDIR)
	if test ! $(SVMLIGHT) = yes; then $(REMOVE_SVMLIGHT); fi

	# remove top level makefile from distribution
	rm -f $(DESTDIR)/src/.authors
	cp -f src/lib/versionstring.h $(DESTDIR)/src/lib/

svn-ignores: .svn_ignores
	find . -name .svn -prune -o -type d -exec svn propset svn:ignore -F .svn_ignores {} \;

clean:
	rm -rf $(DESTDIR)

distclean:
	make -C src distclean
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
