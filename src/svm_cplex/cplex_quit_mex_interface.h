/*
 * MATLAB Compiler: 2.1
 * Date: Sun Nov 11 18:41:15 2001
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

#ifndef __cplex_quit_mex_interface_h
#define __cplex_quit_mex_interface_h 1

#ifdef __cplusplus
extern "C" {
#endif

#include "libmatlb.h"

extern void InitializeModule_cplex_quit_mex_interface(void);
extern void TerminateModule_cplex_quit_mex_interface(void);
extern _mexLocalFunctionTable _local_function_table_cplex_quit;

extern mxArray * mlfNCplex_quit(int nargout, mlfVarargoutList * varargout, ...);
extern mxArray * mlfCplex_quit(mlfVarargoutList * varargout, ...);
extern void mlfVCplex_quit(mxArray * synthetic_varargin_argument, ...);
extern void mlxCplex_quit(int nlhs,
                          mxArray * plhs[],
                          int nrhs,
                          mxArray * prhs[]);

#ifdef __cplusplus
}
#endif

#endif
