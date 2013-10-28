.PHONY: examples

examples:
	bash ./generate_documented.sh

distclean: clean

clean:
	(cd documented && rm -rf cmdline_static elwms_static libshogun matlab_and_octave \
		octave_modular python_static python_modular r_static r_modular csharp_modular \
		lua_modular ruby_modular java_modular )
	$(MAKE) -C undocumented/libshogun clean
	rm -f missing.log undocumented/*/*.txt
