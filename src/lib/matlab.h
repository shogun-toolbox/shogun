#ifdef HAVE_MATLAB
//use compatibility mode w/ matlab 7.4 (size_t is int)
#define MX_COMPAT_32 1
#include <mex.h>
#endif //HAVE_MATLAB
