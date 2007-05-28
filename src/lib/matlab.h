#ifdef HAVE_MATLAB

#include <mex.h>

//use compatibility mode w/ matlab <7.x
#if !defined(MX_API_VER) || MX_API_VER<0x07040000
#define mwSize INT
#define mwIndex INT
#endif

#endif //HAVE_MATLAB
