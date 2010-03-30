.PHONY: examples

examples:
	bash ./generate_documented.sh

distclean: clean

clean: 
	(cd documented && rm -rf cmdline elwms libshogun matlab_and_octave \
		octave_modular python python_modular r r_modular )
	$(MAKE) -C undocumented/libshogun clean
	rm -f missing.log undocumented/*/*.txt
