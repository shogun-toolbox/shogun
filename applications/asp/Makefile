release-dir := msplicer-0.3

release:
	( cd .. ; mkdir -p $(release-dir)/data ; \
	cp python/msplicer python/LICENSE python/README python/NEWS python/*.py $(release-dir) ; \
	tar cjvf $(release-dir).tar.bz2 $(release-dir) )
clean:
	rm -f *.pyc
