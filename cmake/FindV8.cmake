# Locate V8
# This module defines
# V8_LIBRARY
# V8_FOUND, if false, do not try to link to V8
# V8_INCLUDE_DIR, where to find the headers

FIND_PATH(V8_INCLUDE_DIR v8.h
    ${V8_DIR}/include
    $ENV{V8_DIR}/include
    $ENV{V8_DIR}
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/include
    /usr/include
    /sw/include # Fink
    /opt/local/include # DarwinPorts
    /opt/csw/include # Blastwave
    /opt/include
    /usr/freeware/include
    /devel
)

# On non-Unix platforms (Mac and Windows specifically based on the forum),
# V8 builds separate shared (or at least linkable) libraries for v8_base and v8_snapshot
IF(NOT UNIX)
    FIND_LIBRARY(V8_BASE_LIBRARY
        NAMES v8_base v8_base.ia32 v8_base.x64 libv8_base
        PATHS
        ${V8_DIR}
        ${V8_DIR}/lib
        ${V8_DIR}/build/Release/lib
        $ENV{V8_DIR}
        $ENV{V8_DIR}/lib
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/lib
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        /usr/freeware/lib64
    )

    FIND_LIBRARY(V8_SNAPSHOT_LIBRARY
        NAMES v8_snapshot libv8_snapshot
        PATHS
        ${V8_DIR}
        ${V8_DIR}/lib
        ${V8_DIR}/build/Release/lib
        $ENV{V8_DIR}
        $ENV{V8_DIR}/lib
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/lib
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        /usr/freeware/lib64
    )

    FIND_LIBRARY(V8_SNAPSHOT_LIBRARY_DEBUG
        NAMES v8_snapshot libv8_snapshot
        PATHS
        ${V8_DIR}
        ${V8_DIR}/lib
        ${V8_DIR}/build/Debug/lib
        $ENV{V8_DIR}
        $ENV{V8_DIR}/lib
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/lib
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        /usr/freeware/lib64
    )

# On Linux, there is just a libv8.so shared library built.
# (well, there are pseudo-static libraries libv8_base.a and libv8_snapshot.a
# but they don't seem to link correctly)
ELSE()
    FIND_LIBRARY(V8_LIBRARY
        NAMES v8
        PATHS
        ${V8_DIR}
        ${V8_DIR}/lib
        ${V8_DIR}/build/Release/lib
        # Having both architectures listed is problematic if both have been
        # built (which is the default)
        ${V8_DIR}/out/ia32.release/lib.target/
        ${V8_DIR}/out/x64.release/lib.target/
        $ENV{V8_DIR}
        $ENV{V8_DIR}/lib
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/lib
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        /usr/freeware/lib64
    )
ENDIF(NOT UNIX)

include(FindPackageHandleStandardArgs)
IF(NOT UNIX)
    find_package_handle_standard_args(V8 REQUIRED_VARS V8_BASE_LIBRARY V8_SNAPSHOT_LIBRARY V8_INCLUDE_DIR)
ELSE()
    find_package_handle_standard_args(V8 REQUIRED_VARS V8_LIBRARY V8_INCLUDE_DIR)
ENDIF()
