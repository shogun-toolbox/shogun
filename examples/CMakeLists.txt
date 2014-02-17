IF (NOT TRAVIS_DISABLE_LIBSHOGUN_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/libshogun)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/libshogun)
ENDIF()

IF (PythonModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/python_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/python_modular)
ENDIF()

# lua modular
IF (LuaModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/lua_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/lua_modular)
ENDIF()

# java modular
IF (JavaModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/java_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/java_modular)
ENDIF()

# ruby modular
IF (RubyModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/ruby_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/ruby_modular)
ENDIF()

# octave modular
IF (OctaveModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/octave_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/octave_modular)
ENDIF()

# csharp modular
IF (CSharpModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/csharp_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/csharp_modular)
ENDIF()

# r modular
IF (RModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/r_modular)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/r_modular)
ENDIF()

# perl modular
IF (PerlModular AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/perl_modular)
	#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/perl_modular)
ENDIF()

IF (MatlabStatic AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/matlab_static)
	#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/matlab_static)
ENDIF()

IF (CmdLineStatic AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/undocumented/cmdline_static)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/undocumented/cmdline_static)
ENDIF()

