IF(BUILD_COOKBOOK)
    add_subdirectory(cookbook)
    LIST(APPEND DOC_DEPENDENCIES cookbook)
ENDIF()

IF(DOXYGEN_FOUND)
    add_subdirectory(doxygen)
    LIST(APPEND DOC_DEPENDENCIES doxygen)
ENDIF()

add_custom_target(doc DEPENDS ${DOC_DEPENDENCIES})
