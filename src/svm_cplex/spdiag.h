/*
 * MATLAB Compiler: 2.1
 * Date: Sun Nov 11 18:46:58 2001
 * Arguments: "-B" "macro_default" "-O" "all" "-O" "fold_scalar_mxarrays:on"
 * "-O" "fold_non_scalar_mxarrays:on" "-O" "optimize_integer_for_loops:on" "-O"
 * "array_indexing:on" "-O" "optimize_conditionals:on" "-m" "-W" "main" "-L"
 * "C" "-t" "-T" "link:exe" "-h" "libmmfile.mlib" "-O" "all" "-O"
 * "fold_scalar_mxarrays:on" "-O" "fold_non_scalar_mxarrays:on" "-O"
 * "optimize_integer_for_loops:on" "-O" "array_indexing:on" "-O"
 * "optimize_conditionals:on" "train_svm" 
 */

#ifndef MLF_V2
#define MLF_V2 1
#endif

#ifndef __spdiag_h
#define __spdiag_h 1

#ifdef __cplusplus
extern "C" {
#endif

#include "libmatlb.h"

extern void InitializeModule_spdiag(void);
extern void TerminateModule_spdiag(void);
extern _mexLocalFunctionTable _local_function_table_spdiag;

extern mxArray * mlfSpdiag(mxArray * diagonal);
extern void mlxSpdiag(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]);

#ifdef __cplusplus
}
#endif

#endif
