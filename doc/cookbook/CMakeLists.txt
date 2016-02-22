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

add_custom_target(shogun_cookbook_sphinx_extensions
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/extensions"
    "${BINARY_BUILD_DIR}"
    COMMENT "Copying cookbook Sphinx extensions to ${BINARY_BUILD_DIR}")

add_custom_target(shogun_cookbook_sphinx_templates
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/source/templates"
    "${BINARY_BUILD_DIR}/templates"
    COMMENT "Copying cookbook Sphinx templates to ${BINARY_BUILD_DIR}")

add_custom_target(shogun_cookbook_sphinx_static
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/source/static"
    "${BINARY_BUILD_DIR}/static"
    COMMENT "Copying cookbook Sphinx static overrides to ${BINARY_BUILD_DIR}")
    
add_custom_target(shogun_cookbook_sphinx_linkcheck
    ${SPHINX_EXECUTABLE}
    -q -b linkcheck
    -c "${BINARY_BUILD_DIR}"
    -d "${SPHINX_CACHE_DIR}"
    -D generated_examples_path="${CMAKE_BINARY_DIR}/examples/meta/"
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${SPHINX_HTML_DIR}"
    COMMENT "Cookbook Sphinx validating external links"
    DEPENDS shogun_cookbook_sphinx_extensions shogun_cookbook_sphinx_templates
            shogun_cookbook_sphinx_static shogun_meta_examples)

add_custom_target(shogun_cookbook ALL
    ${SPHINX_EXECUTABLE}
    -q -b html
    -c "${BINARY_BUILD_DIR}"
    -d "${SPHINX_CACHE_DIR}"
    -D generated_examples_path="${CMAKE_BINARY_DIR}/examples/meta/"
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${SPHINX_HTML_DIR}"
    COMMENT "Cookbook Sphinx building HTML"
    DEPENDS shogun_cookbook_sphinx_extensions shogun_cookbook_sphinx_templates
            shogun_cookbook_sphinx_static shogun_meta_examples
            shogun_cookbook_sphinx_linkcheck)
