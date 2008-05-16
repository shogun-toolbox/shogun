# Usage scenarios
#
# * To make a release (and tag it) run (run make distclean before!)
#
#		(cd trunk/src;  rm ChangeLog ; make ChangeLog ; svn ci -m "updated changelog")
#       make svn-tag-release  
#       (cd releases/shogun_0.6.2)
#       make release
#       make update-webpage
#
# * To create a debian .orig.tar.gz run
#
#       make vanilla-package DEBIAN=yes
#
# * To create a snapshot run
#
#       make vanilla-package SNAPSHOT=yes
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
all: doc release matlab python octave r
endif


.PHONY: doc release matlab python octave r vanilla-package r-package

DESTDIR := ../$(RELEASENAME)
REMOVE_SVMLIGHT := rm -f $(DESTDIR)/src/classifier/svm/SVM_light.* $(DESTDIR)/src/classifier/svm/Optimizer.* $(DESTDIR)/src/regression/svr/SVR_light.* $(DESTDIR)/src/LICENSE.SVMlight; \
rm -f $(DESTDIR)/testsuite/data/classifier/SVMLight* $(DESTDIR)/testsuite/data/regression/SVRLight*	; \
grep -rl USE_SVMLIGHT $(DESTDIR)| xargs --no-run-if-empty sed -i '/\#ifdef USE_SVMLIGHT/,/\#endif \/\/USE_SVMLIGHT/c \\' ; \
sed -i '/^ \* EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS/,/ \* this program is free software/c\ * This program is free software; you can redistribute it and/or modify' $(DESTDIR)/src/kernel/Kernel.cpp $(DESTDIR)/src/kernel/Kernel.h ; \
sed -i '/^SVMlight:$$/,/^$$/c\\' $(DESTDIR)/src/LICENSE

# We assume that a release is always created from a SVN working copy.

release: src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h vanilla-package r-package
	rm -rf $(DESTDIR)

vanilla-package: src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	rm -f $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	$(COMPRESS) -9 $(DESTDIR).tar

# build r-package
r-package:	src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h
	-make -C $(DESTDIR)/r package
	cp $(DESTDIR)/r/*.tar.gz ../

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
	md5sum ../sg_$(MAINVERSION).tar.gz >../sg_$(MAINVERSION).md5sum
	gpg --sign $(DESTDIR).tar.bz2
	gpg --sign ../sg_$(MAINVERSION).tar.gz

	ssh vserver mkdir -m 0755 -p /pub/shogun-ftp/releases/$(VERSIONBASE)/sources \
		/pub/shogun-ftp/releases/$(VERSIONBASE)/Rsources
	scp ../sg_$(MAINVERSION).tar.gz ../sg_$(MAINVERSION).tar.gz.gpg \
		../sg_$(MAINVERSION).md5sum vserver:/pub/shogun-ftp/releases/$(VERSIONBASE)/Rsources/
	scp $(DESTDIR).tar.bz2 $(DESTDIR).tar.bz2.gpg $(DESTDIR).md5sum \
		../sg_$(MAINVERSION).md5sum vserver:/pub/shogun-ftp/releases/$(VERSIONBASE)/sources/
	ssh vserver chmod 644 /pub/shogun-ftp/releases/$(VERSIONBASE)/sources/*.* \
		/pub/shogun-ftp/releases/$(VERSIONBASE)/Rsources/*.*
	
	rm -rf doc/html
	make -C doc
	ssh vserver rm -f /pub/shogun/doc/*.*
	cd doc/html && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh vserver tar -C /pub/shogun/doc/ -xjvf -
	ssh vserver chmod 644 /pub/shogun/doc/*.*
	ssh vserver ./bin/shogun_doc_install.sh
	rm -rf doc/html

svn-tag-release: src/lib/versionstring.h
	sed -i "s/^Version.*$$/Version: $(MAINVERSION)/" r/sg/DESCRIPTION
	sed -i "s/^Date:.*$$/Date: `date +%Y-%m-%d`/" r/sg/DESCRIPTION
	sed -i "s/^SHOGUN:=.*$$/SHOGUN:=sg_$(MAINVERSION)-1.tar.gz/" r/Makefile
	sed -i 's/VERSION_RELEASE "svn/VERSION_RELEASE "v$(MAINVERSION)/' src/lib/versionstring.h
	sed -i "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = v$(MAINVERSION)/" doc/Doxyfile
	svn ci -m "Preparing for new Release shogun_$(MAINVERSION)"
	-cd .. && svn --force rm releases/shogun_$(MAINVERSION)
	-cd .. && svn commit releases -m "clean old tag"
	cd .. && svn cp trunk releases/shogun_$(MAINVERSION)
	cp src/lib/versionstring.h ../releases/shogun_$(MAINVERSION)/src/lib/versionstring.h
	sed -i "s| lib/versionstring.h||" ../releases/shogun_$(MAINVERSION)/src/Makefile
	cd ../releases && svn add shogun_$(MAINVERSION)/src/lib/versionstring.h
	cd ../releases && svn ci -m "Tagging shogun_$(MAINVERSION) release"

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
	rm -rf $(DESTDIR)/r/sg/src
	cp -f src/lib/versionstring.h $(DESTDIR)/src/lib/

svn-ignores: .svn_ignores
	find . -name .svn -prune -o -type d -exec svn propset svn:ignore -F .svn_ignores {} \;

clean:
	rm -rf $(DESTDIR)

distclean:
	make -C src distclean
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
