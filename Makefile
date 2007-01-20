# You can execute
#
# make release COMPRESS=gzip EXTRAVERSION=+svn20061202
#
# to use gzip instead of bzip2 and to append an extra version string.
# 
# the following additional options may be set
#
# DEBIAN=yes   						-> use debian naming scheme shogun_0.1.1+svn1337.orig.tar.gz
# SVMLIGHT=no						-> remove svm light
# COMPRESS=cruncher					-> use cruncher (bz2/gz etc as file compressor)
# SNAPSHOT=yes						-> use svn snapshot naming scheme
# MAINVERSION=0.2.0					-> main version
# EXTRAVERSION=+svn20061202			-> extra version string
# RELEASENAME=shogun-3.0.0+extra	-> use different releasename, here shogun-3.0.0+extra
#
# for example make DEBIAN=yes SNAPSHOT=yes
#


DEBIAN := no
SVMLIGHT := yes
COMPRESS := bzip2
MAINVERSION := 0.2.1
EXTRAVERSION := 
RELEASENAME := shogun-$(MAINVERSION)$(EXTRAVERSION)
SVNVERSION = $(shell svn info | grep Revision: | cut -d ' ' -f 2)

ifeq ($(DEBIAN),yes)
RELEASENAME := shogun_$(MAINVERSION)
ifeq ($(SNAPSHOT),yes)
RELEASENAME := $(RELEASENAME)+svn$(SVNVERSION)
endif
RELEASENAME := $(RELEASENAME).orig
COMPRESS := gzip
SVMLIGHT := no
TARGET := vanilla-package
else
all: doc release matlab python octave R
endif

.PHONY: doc release matlab python octave R vanilla-package r-package

DESTDIR := ../$(RELEASENAME)
REMOVE_SVMLIGHT := rm -f $(DESTDIR)/src/classifier/svm/{SVM_light,Optimizer}* $(DESTDIR)/src/regression/svr/SVR_light.* $(DESTDIR)/src/LICENSE.SVMlight; \
grep -rl USE_SVMLIGHT $(DESTDIR)| xargs --no-run-if-empty sed -i '/\#ifdef USE_SVMLIGHT/,/\#endif \/\/USE_SVMLIGHT/c \\' ; \
sed -i '/^ \* EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS/,/ \* this program is free software/c\ * This program is free software; you can redistribute it and/or modify' $(DESTDIR)/src/kernel/Kernel.{cpp,h} ; \
sed -i '/^SVMlight:$$/,/^$$/c\\' $(DESTDIR)/src/LICENSE

all: $(TARGET)

src/lib/versionstring.h:
	make -C src lib/versionstring.h

# We assume that a release is always created from a SVN working copy.

release: src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h vanilla-package r-package
	rm -rf $(DESTDIR)

vanilla-package: src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	rm -f $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	$(COMPRESS) -9 $(DESTDIR).tar

# build R-package
r-package:	src/lib/versionstring.h $(DESTDIR)/src/lib/versionstring.h
	cd $(DESTDIR)/R && make package && cp *.tar.gz ../../

$(DESTDIR)/src/lib/versionstring.h: src/lib/versionstring.h
	rm -rf $(DESTDIR)
	svn export . $(DESTDIR)
	if [ ! $(SVMLIGHT) = yes ]; then $(REMOVE_SVMLIGHT); fi

	# remove top level makefile from distribution
	rm -f $(DESTDIR)/Makefile
	rm -f $(DESTDIR)/src/.authors
	mv -f src/lib/versionstring.h $(DESTDIR)/src/lib/

clean:
	rm -rf $(DESTDIR)

mrproper:
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
