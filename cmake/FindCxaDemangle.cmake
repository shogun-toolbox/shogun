# Module that checks whether the compiler supports the
# abi::__cxa_demangle function required to
# make the type names returned by typeid() human-readable
#
# Sets the following variable:
# HAVE_CXA_DEMANGLE
#
# perform tests
include(CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES("#include <typeinfo>
#include <cxxabi.h>
int main(void){
    int foobar = 0;
    const char *foo = typeid(foobar).name();
    int status;
    char *demangled = abi::__cxa_demangle( foo, 0, 0, &status );
}" HAVE_CXA_DEMANGLE)

include (FindPackageHandleStandardArgs)
# prevent useless message from being displayed
set (FIND_PACKAGE_MESSAGE_DETAILS_CxaDemangle "[1][v()]"
  CACHE INTERNAL "Details about finding CxaDemangle")
find_package_handle_standard_args (CxaDemangle DEFAULT_MSG HAVE_CXA_DEMANGLE)

