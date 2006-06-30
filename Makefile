RELEASENAME:=shogun-0.1.2

.PHONY: doc release matlab python octave R

all: doc release matlab python octave R

src/lib/versionstring.h: 
	make -C src lib/versionstring.h

release: src/lib/versionstring.h
	make -C doc distclean 
	cp -p src/lib/versionstring.h versionstring.h
	make -C src distclean 
	mv versionstring.h src/lib/versionstring.h
	make -C R clean 
	rm -rf ../$(RELEASENAME) || true
	rm -f ../$(RELEASENAME).tar.bz2 || true
	cd .. && cp -a trunk $(RELEASENAME) && \
	(find ./$(RELEASENAME) \( -name .svn -o -name '.*.swp' -o -name '.nfs*' -o -name '*~' \) -exec rm -rf \{\} \; 2>/dev/null ; \
	tar cvf $(RELEASENAME).tar $(RELEASENAME) && bzip2 -9 $(RELEASENAME).tar && rm -rf $(RELEASENAME))

