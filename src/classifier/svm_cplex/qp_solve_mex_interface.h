#ifdef MEX
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

#ifndef __qp_solve_mex_interface_h
#define __qp_solve_mex_interface_h 1

#ifdef __cplusplus
extern "C" {
#endif

#include "libmatlb.h"

extern void InitializeModule_qp_solve_mex_interface(void);
extern void TerminateModule_qp_solve_mex_interface(void);
extern _mexLocalFunctionTable _local_function_table_qp_solve;

extern mxArray * mlfNQp_solve(int nargout, mlfVarargoutList * varargout, ...);
extern mxArray * mlfQp_solve(mlfVarargoutList * varargout, ...);
extern void mlfVQp_solve(mxArray * synthetic_varargin_argument, ...);
extern void mlxQp_solve(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]);

#ifdef __cplusplus
}
#endif

#endif
#endif
