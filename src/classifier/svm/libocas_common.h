#ifdef OCAS_MATLAB
#include "mex.h"
#define OCAS_PLUS_INF mxGetInf()
#define OCAS_PRINT(x...) mexPrintf(x)
#define OCAS_CALLOC(x...) mxCalloc(x)
#define OCAS_FREE(x...) mxFree(x)
#define OCAS_ERRORMSG(x...) mexErrMsgTxt(x)
#else
#define OCAS_PLUS_INF (-log(0.0))
#define OCAS_PRINT(x...) printf(x)
#define OCAS_CALLOC(x...) calloc(x)
#define OCAS_FREE(x...) free(x)
#define OCAS_ERRORMSG(x...) { printf(x); return(1); }
#endif

#define INDEX2(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define MAX(A,B) ((A) < (B) ? (B) : (A))
#define ABS(A) ((A) < 0 ? -(A) : (A))

