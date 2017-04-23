#PROJECT(sphinxdoc)
#cmake_minimum_required(VERSION 2.8.8)
include(FindMetaExamples)
find_excluded_cookbook_pages()

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)

add_custom_target(cookbook_copy_source
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${BINARY_BUILD_DIR}"
    COMMENT "Copying cookbook Sphinx source to ${BINARY_BUILD_DIR}")

add_custom_target(cookbook_make_quickstart_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BINARY_BUILD_DIR}/quickstart
    COMMENT "Creating Quickstart folder at ${BINARY_BUILD_DIR}/quickstart")

SET(interface_rst ${BINARY_BUILD_DIR}/quickstart/interfaces.rst)
ADD_CUSTOM_COMMAND(OUTPUT ${interface_rst}
        COMMAND ${PANDOC_EXECUTABLE} -f markdown -t rst -o ${interface_rst}
        ${CMAKE_SOURCE_DIR}/doc/readme/INTERFACES.md
        DEPENDS cookbook_copy_source cookbook_make_quickstart_dir
        WORKING_DIRECTORY ${BINARY_BUILD_DIR})

add_custom_target(cookbook_sphinx_extensions
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/extensions"
    "${BINARY_BUILD_DIR}"
    COMMENT "Copying cookbook Sphinx extensions to ${BINARY_BUILD_DIR}")

add_custom_target(cookbook_sphinx_linkcheck
    ${SPHINX_EXECUTABLE}
    -q -b linkcheck
    -c "${BINARY_BUILD_DIR}"
    -d "${SPHINX_CACHE_DIR}"
    -D generated_examples_path="${CMAKE_BINARY_DIR}/examples/meta/"
    -D exclude_patterns="${EXCLUDED_COOKBOOK_PAGES}"
    "${BINARY_BUILD_DIR}"
    "${SPHINX_HTML_DIR}"
    COMMENT "Cookbook Sphinx validating external links"
    DEPENDS cookbook_copy_source cookbook_sphinx_extensions meta_examples)

add_custom_target(cookbook
    ${SPHINX_EXECUTABLE}
    -q -b html
    -c "${BINARY_BUILD_DIR}"
    -d "${SPHINX_CACHE_DIR}"
    -D generated_examples_path="${CMAKE_BINARY_DIR}/examples/meta/"
    -D exclude_patterns="${EXCLUDED_COOKBOOK_PAGES}"
    "${BINARY_BUILD_DIR}"
    "${SPHINX_HTML_DIR}"
    COMMENT "Cookbook Sphinx building HTML"
    DEPENDS cookbook_copy_source cookbook_sphinx_extensions meta_examples
            ${interface_rst})
