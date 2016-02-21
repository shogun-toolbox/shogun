#PROJECT(sphinxdoc)
#cmake_minimum_required(VERSION 2.8.8)

set(SPHINX_EXECUTABLE "sphinx-build")

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/source/conf.py"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)

add_custom_target(shogun_sphinx_doc_extensions
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/extensions"
    "${BINARY_BUILD_DIR}"
    COMMENT "Copying Sphinx extensions to ${BINARY_BUILD_DIR}")

add_custom_target(shogun_sphinx_doc_templates
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/source/templates"
    "${BINARY_BUILD_DIR}/templates"
    COMMENT "Copying Sphinx templates to ${BINARY_BUILD_DIR}")

add_custom_target(shogun_sphinx_doc_static
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/source/static"
    "${BINARY_BUILD_DIR}/static"
    COMMENT "Copying Sphinx static overrides to ${BINARY_BUILD_DIR}")
    
add_custom_target(shogun_sphinx_doc_linkcheck
    ${SPHINX_EXECUTABLE}
    -q -b linkcheck
    -c "${BINARY_BUILD_DIR}"
    -d "${SPHINX_CACHE_DIR}"
    -D generated_examples_path="${CMAKE_BINARY_DIR}/examples/meta/"
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${SPHINX_HTML_DIR}"
    COMMENT "Sphinx validating external links"
    DEPENDS shogun_sphinx_doc_extensions shogun_sphinx_doc_templates
            shogun_sphinx_doc_static shogun_meta_examples)

add_custom_target(shogun_sphinx_doc ALL
    ${SPHINX_EXECUTABLE}
    -q -b html
    -c "${BINARY_BUILD_DIR}"
    -d "${SPHINX_CACHE_DIR}"
    -D generated_examples_path="${CMAKE_BINARY_DIR}/examples/meta/"
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${SPHINX_HTML_DIR}"
    COMMENT "Sphinx building HTML cookbook"
    DEPENDS shogun_sphinx_doc_extensions shogun_sphinx_doc_templates
            shogun_sphinx_doc_static shogun_meta_examples
            shogun_sphinx_doc_linkcheck)
