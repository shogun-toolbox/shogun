/*
 * MATLAB Compiler: 2.1
 * Date: Mon Jan 28 12:05:31 2002
 * Arguments: "-B" "macro_default" "-O" "all" "-O" "fold_scalar_mxarrays:on"
 * "-O" "fold_non_scalar_mxarrays:on" "-O" "optimize_integer_for_loops:on" "-O"
 * "array_indexing:on" "-O" "optimize_conditionals:on" "-m" "-W" "main" "-L"
 * "C" "-t" "-T" "link:exe" "-h" "libmmfile.mlib" "cleaner" 
 */

#ifdef MATLAB
#ifndef MLF_V2
#define MLF_V2 1
#endif

#ifndef __cleaner_h
#define __cleaner_h 1

#ifdef __cplusplus
extern "C" {
#endif

#include "libmatlb.h"

extern void InitializeModule_cleaner(void);
extern void TerminateModule_cleaner(void);
extern _mexLocalFunctionTable _local_function_table_cleaner;

extern mxArray * mlfCleaner(mxArray * covz, mxArray * thresh);
extern void mlxCleaner(INT nlhs, mxArray * plhs[], INT nrhs, mxArray * prhs[]);

#ifdef __cplusplus
}
#endif

#endif

#endif
