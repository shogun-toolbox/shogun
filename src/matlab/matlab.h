#include <mex.h>

//use compatibility mode w/ matlab <7.x
#if !defined(MX_API_VER) || MX_API_VER<0x07040000
#define mwSize int32_t
#define mwIndex int32_t

#define mxIsLogicalScalar(x) false
#define mxIsLogicalScalarTrue(x) false
#endif
