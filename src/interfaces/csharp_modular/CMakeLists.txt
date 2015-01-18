include(CommonModularInterface)

GENERATE_MODULAR_TARGET(csharp ${CMAKE_CURRENT_SOURCE_DIR} "")

ADD_CUSTOM_COMMAND(TARGET csharp_modular
				   POST_BUILD
				   COMMAND "${CSHARP_COMPILER}" /t:library *.cs /out:modshogun.dll
				   COMMENT "Creating modshogun.dll")

SET(CSHARP_MODULAR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
INSTALL(TARGETS csharp_modular
	LIBRARY DESTINATION lib/cli/shogun
	COMPONENT csharp)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/modshogun.dll
		DESTINATION lib/cli/shogun
		COMPONENT csharp)
