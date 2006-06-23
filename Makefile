RELEASENAME:=shogun-0.1.1

.PHONY: doc release matlab python octave R

all: doc release matlab python octave R

release:
	make -C doc distclean 
	make -C src distclean 
	make -C R clean 
	rm -rf ../$(RELEASENAME) || true
	rm -f ../$(RELEASENAME).tar.bz2 || true
	cd .. && cp -a trunk $(RELEASENAME) && \
	(find ./$(RELEASENAME) \( -name .svn -o -name '.*.swp' -o -name '.nfs*' -o -name '*~' \) -exec rm -rf \{\} \; 2>/dev/null ; \
	tar cvf $(RELEASENAME).tar $(RELEASENAME) && bzip2 -9 $(RELEASENAME).tar && rm -rf $(RELEASENAME))

