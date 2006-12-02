MAINVERSION := 0.1.2
#EXTRAVERSION := +svn20061202
COMPRESS := bzip2

.PHONY: doc release matlab python octave R

all: doc release matlab python octave R

# You can execute
#
# make release COMPRESS=gzip EXTRAVERSION=+svn20061202
#
# to use gzip instead of bzip2 and to append an extra version string.

RELEASENAME := shogun-$(MAINVERSION)$(EXTRAVERSION)
DESTDIR := ../$(RELEASENAME)

src/lib/versionstring.h:
	make -C src lib/versionstring.h

# We assume that a release is always created from a SVN working copy.
release: src/lib/versionstring.h
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	svn export . $(DESTDIR)
	mv -f src/lib/versionstring.h $(DESTDIR)/src/lib/
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	$(COMPRESS) -9 $(DESTDIR).tar
	rm -rf $(DESTDIR)
