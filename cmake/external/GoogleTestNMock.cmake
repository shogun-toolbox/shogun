MergeCFLAGS()
GetCompilers()

# TODO: set -fPIC only if needed
SET(MERGED_CXX_FLAGS "${MERGED_CXX_FLAGS} -fPIC")

SET (GMOCK_REVISION 443)
include(ExternalProject)
ExternalProject_Add(
	GoogleMock
	SVN_REPOSITORY http://googlemock.googlecode.com/svn/trunk
	SVN_REVISION -r ${GMOCK_REVISION}
	UPDATE_COMMAND ""
    TIMEOUT 10
	PREFIX ${CMAKE_BINARY_DIR}/GoogleMock
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/GoogleMock
	SOURCE_DIR ${THIRD_PARTY_DIR}/GoogleMock
	INSTALL_COMMAND ""
	CMAKE_ARGS 	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${THIRD_PARTY_DIR}/libs/gmock
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${THIRD_PARTY_DIR}/libs/gmock
		-DCMAKE_CXX_FLAGS:STRING=${MERGED_CXX_FLAGS}${CMAKE_DEFINITIONS}
		-DCMAKE_C_COMPILER:STRING=${C_COMPILER}
		-DCMAKE_CXX_COMPILER:STRING=${CXX_COMPILER}
)

ExternalProject_Get_Property(GoogleMock STAMP_DIR)
ExternalProject_Get_Property(GoogleMock SOURCE_DIR)

ExternalProject_Add_Step(
	GoogleMock check_revision
	COMMAND ${CMAKE_COMMAND} -D STAMP_DIR=${STAMP_DIR}
                     		 -D SRC_DIR=${SOURCE_DIR}
                     		 -D SVN_EXEC=${Subversion_SVN_EXECUTABLE}
                     		 -D REVISION=${GMOCK_REVISION}
                     		 -D PROJECT_NAME=GoogleMock
                     		 -P ${CMAKE_MODULE_PATH}/CheckSVNRevision.cmake
	COMMENT "Checking Google Mock checked out revision"
	DEPENDERS download
	DEPENDEES mkdir
)

UNSET(C_COMPILER)
UNSET(CXX_COMPILER)