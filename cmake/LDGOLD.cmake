include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(ENABLE_LDGOLD
    "Use GNU gold linker" ON
    "NOT WIN32;NOT APPLE" OFF)

GetCompilers()

set(LDGOLD_FOUND FALSE)
if(ENABLE_LDGOLD)
  execute_process(COMMAND ${C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)
  if(LD_VERSION MATCHES "GNU gold")
    set(LDGOLD_FOUND TRUE)
    message(STATUS "Linker: GNU gold")
  else()
    message(WARNING "GNU gold linker is not available, falling back to default system linker")
  endif()
else()
  message(STATUS "Linker: Default system linker")
endif()

set(DEFAULT_ENABLE_DEBUGFISSION FALSE)
if(CMAKE_BUILD_TYPE STREQUAL Debug AND LDGOLD_FOUND)
    if (CCACHE_FOUND AND CCACHE_VERSION VERSION_LESS 3.2.3)
        # only ccache 3.2.3 and later supports DEBUGFISSION
        # i.e. -gsplit-dwarf flag. see 
        # https://bugzilla.samba.org/show_bug.cgi?id=10005
        MESSAGE(WARNING "Debug Fission is available but your ccache does not supports it")
    else()
        set(DEFAULT_ENABLE_DEBUGFISSION TRUE)
    endif()
endif()

include(CMakeDependentOption)
cmake_dependent_option(ENABLE_DEBUGFISSION
    "Enable Debug Fission support" ON
    "DEFAULT_ENABLE_DEBUGFISSION;" OFF)

set(DEBUGFISSION_FOUND FALSE)
if(ENABLE_DEBUGFISSION)
  include(TestCXXAcceptsFlag)
  check_cxx_accepts_flag(-gsplit-dwarf CXX_ACCEPTS_GSPLIT_DWARF)
  if(CXX_ACCEPTS_GSPLIT_DWARF)
    set(DEBUGFISSION_FOUND TRUE)
    message(STATUS "Debug Fission enabled")
  else()
    message(WARNING "Debug Fission is not available")
  endif()
endif()

function(SET_LDGOLD)
    foreach(t ${ARGN})
        if (TARGET ${t})
            get_target_property(TARGET_TYPE ${t} TYPE)

            if (${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
                target_link_libraries(${t} INTERFACE -fuse-ld=gold)
                if (ENABLE_DEBUGFISSION)
                    target_compile_options(${t} INTERFACE -gsplit-dwarf)
                    target_link_libraries(${t} INTERFACE -Wl,--gdb-index)
                endif()
            else()
                if (ENABLE_DEBUGFISSION)
                    target_compile_options(${t} PRIVATE -gsplit-dwarf)
                endif()

                if (NOT ${TARGET_TYPE} STREQUAL OBJECT_LIBRARY)
                    target_link_libraries(${t} PRIVATE -fuse-ld=gold)
                    if (ENABLE_DEBUGFISSION)
                        target_link_libraries(${t} PRIVATE -Wl,--gdb-index)
                    endif()
                endif()
            endif()
        endif()
    endforeach()
endfunction()
