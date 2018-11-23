SET(VARIANT_RELEASE_VERSION v1.3.0)
SET(VARIANT_SOURCE_DIR ${THIRD_PARTY_DIR}/variant)
SET(VARIANT_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR}/variant)
include(ExternalProject)
ExternalProject_Add(
		variant
		PREFIX ${CMAKE_BINARY_DIR}/variant
		SOURCE_DIR ${VARIANT_SOURCE_DIR}
		GIT_REPOSITORY https://github.com/mpark/variant.git
		GIT_TAG caaff9a8cb00b7897ee581e2916e78a6bb916b20
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${VARIANT_SOURCE_DIR}/include/mpark ${VARIANT_INCLUDE_DIR}
)

add_dependencies(libshogun variant)
