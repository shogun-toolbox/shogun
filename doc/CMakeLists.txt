# cookbook and dependencies (note there are in fact more, see its requirements.txt)
FIND_PACKAGE(Sphinx)
FIND_PACKAGE(Pandoc)
IF(BUILD_META_EXAMPLES AND SPHINX_FOUND AND PANDOC_FOUND)
    add_subdirectory(cookbook)
    LIST(APPEND DOC_DEPENDENCIES cookbook)
ENDIF()

IF(DOXYGEN_FOUND)
    add_subdirectory(doxygen)
    LIST(APPEND DOC_DEPENDENCIES doxygen)
ENDIF()

add_custom_target(doc DEPENDS ${DOC_DEPENDENCIES})
