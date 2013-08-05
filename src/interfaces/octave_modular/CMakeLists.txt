include(CommonModularInterface)
include_directories(${OCTAVE_INCLUDE_DIRS})

GENERATE_MODULAR_TARGET(octave ${CMAKE_CURRENT_SOURCE_DIR} ${OCTAVE_LIBRARIES})

install (TARGETS octave_modular
    DESTINATION ${OCTAVE_OCT_LOCAL_API_FILE_DIR}/shogun
	COMPONENT octave)

file(WRITE init_shogun.m "%Initialize shogun by loading all of its modules")

#	for i in $(SWIGLIBFILES) ; do echo $$i | sed -e 's=.*/==' -e 's/\.oct/;/' \
#		>>$(DESTDIR)$(OCTDIR)/shogun/init_shogun.m; done
#FOREACH()
#	file(APPEND init_shogun.m "")
#ENDFOREACH()

INSTALL(FILES init_shogun.m
    DESTINATION ${OCTAVE_OCT_LOCAL_API_FILE_DIR}/shogun
	COMPONENT octave)

set(CPACK_COMPONENT_OCTAVE_DISPLAY_NAME "OCTAVE interface" PARENT_SCOPE)
set(CPACK_COMPONENT_OCTAVE_DESCRIPTION "OCTAVE modular interface of Shogun" PARENT_SCOPE)
set(CPACK_COMPONENT_OCTAVE_GROUP "Runtime" PARENT_SCOPE)
set(CPACK_COMPONENT_OCTAVE_DEPENDS libraries PARENT_SCOPE)
SET(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} octave PARENT_SCOPE)
