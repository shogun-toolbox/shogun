IF (NOT TRAVIS_DISABLE_LIBSHOGUN_TESTS)
	add_subdirectory(undocumented/libshogun)
ENDIF()

IF (PythonModular)
	add_subdirectory(undocumented/python_modular)
ENDIF()

# lua modular
IF (LuaModular)
	add_subdirectory(undocumented/lua_modular)
ENDIF()

# java modular
IF (JavaModular)
	add_subdirectory(undocumented/java_modular)
ENDIF()

# ruby modular
IF (RubyModular)
	add_subdirectory(undocumented/ruby_modular)
ENDIF()

# octave modular
IF (OctaveModular)		
	add_subdirectory(undocumented/octave_modular)
ENDIF()

# csharp modular
IF (CSharpModular)
	add_subdirectory(undocumented/csharp_modular)
ENDIF()

# r modular
IF (RModular)
	#add_subdirectory(undocumented/r_modular)
ENDIF()

# perl modular
IF (PerlModular)
	#add_subdirectory(undocumented/perl_modular)
ENDIF()

IF (MatlabStatic)
	#add_subdirectory(undocumented/matlab_static)
ENDIF()

IF (CmdLineStatic)
	add_subdirectory(undocumented/cmdline_static)
ENDIF()

