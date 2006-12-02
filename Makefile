#EXTRAVERSION := +svn20061202
RELEASENAME := shogun-0.1.2$(EXTRAVERSION)
COMPRESS := bzip2

.PHONY: doc release matlab python octave R

all: doc release matlab python octave R

# You can execute
#
# make release COMPRESS=gzip EXTRAVERSION=+svn20061202
#
# to use gzip instead of bzip2 and to append an extra version string.

CURDIR := $(shell pwd)
DESTDIR := $(CURDIR)/../$(RELEASENAME)
FINDFILTER := \( -name .svn -o -name '.*.swp' -o -name '.nfs*' -o -name '*~' \)

release:
	make -C doc distclean 
	make -C src distclean 
	make -C R clean 
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	cp -rl $(CURDIR) $(DESTDIR)
	make -C $(DESTDIR)/src lib/versionstring.h
	find $(DESTDIR) $(FINDFILTER) -print0 | xargs -0 rm -rf
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	$(COMPRESS) -9 $(DESTDIR).tar
	rm -rf $(DESTDIR)

