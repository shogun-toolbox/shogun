#ifdef HAVE_MATLAB

#include <mex.h>
#include "matrix.h"

//use compatibility mode w/ matlab <7.x
#ifndef mwSize
#define mwSize INT
#endif

#ifndef mwIndex
#define mwIndex INT
#endif

#endif //HAVE_MATLAB
