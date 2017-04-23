set (PREFIX ${CMAKE_BINARY_DIR}/MSIntTypes)
SET (MSINTTYPES_COMMIT f9e7c5758ed9e3b9f4b2394de1881c704dd79de0)
include(ExternalProject)
ExternalProject_Add(
	MSIntTypes
	GIT_REPOSITORY https://github.com/chemeris/msinttypes.git
	GIT_TAG ${MSINTTYPES_COMMIT}
	UPDATE_COMMAND ""
    TIMEOUT 10
	PREFIX ${PREFIX}
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/MSIntTypes
	SOURCE_DIR ${THIRD_PARTY_DIR}/MSIntTypes
	INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${PREFIX}/src/MSIntTypes ${THIRD_PARTY_INCLUDE_DIR}/MSIntTypes
	BUILD_COMMAND ""
	CONFIGURE_COMMAND ""
)

ExternalProject_Get_Property(MSIntTypes SOURCE_DIR)
SET(MSINTTYPES_INCLUDE_DIR ${SOURCE_DIR})
add_dependencies(libshogun MSIntTypes)
