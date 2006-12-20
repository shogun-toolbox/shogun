MAINVERSION := 0.2.1
#EXTRAVERSION := +svn20061202
COMPRESS := bzip2
SVMLIGHT := yes

.PHONY: doc release matlab python octave R

all: doc release matlab python octave R

# You can execute
#
# make release COMPRESS=gzip EXTRAVERSION=+svn20061202
#
# to use gzip instead of bzip2 and to append an extra version string.

.PHONY: release vanilla-package r-package

RELEASENAME := shogun-$(MAINVERSION)$(EXTRAVERSION)
DESTDIR := ../$(RELEASENAME)
REMOVE_SVMLIGHT := find $(DESTDIR) -iname '*svm*light*' | xargs rm -f

src/lib/versionstring.h:
	make -C src lib/versionstring.h

# We assume that a release is always created from a SVN working copy.

release: src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h vanilla-package r-package

vanilla-package: src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	$(COMPRESS) -9 $(DESTDIR).tar

r-package:	src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h
	#build R package
	cd $(DESTDIR)/R && make package && cp *.tar.gz ../../
	rm -rf $(DESTDIR)

$(DESTDIR)/src/lib/versionstring.h:
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	svn export . $(DESTDIR)
	if [ ! $(SVMLIGHT) = yes ]; then $(REMOVE_SVMLIGHT); fi

	# remove top level makefile from distribution
	rm -f $(DESTDIR)/Makefile

	# FIXME: This is a hack because .generate_link_dependencies.py is buggy
	# and should better be replaced by 'swig -MM':
	touch $(DESTDIR)/src/classifier/svm/SVM_light.i
	
	mv -f src/lib/versionstring.h $(DESTDIR)/src/lib/
