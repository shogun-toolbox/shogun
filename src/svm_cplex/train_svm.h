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

#ifndef __train_svm_h
#define __train_svm_h 1

#ifdef __cplusplus
extern "C" {
#endif

#include "libmatlb.h"

extern void InitializeModule_train_svm(void);
extern void TerminateModule_train_svm(void);
extern _mexLocalFunctionTable _local_function_table_train_svm;

extern mxArray * mlfTrain_svm(mxArray * * b,
                              mxArray * XT,
                              mxArray * LT,
                              mxArray * C);
extern void mlxTrain_svm(int nlhs,
                         mxArray * plhs[],
                         int nrhs,
                         mxArray * prhs[]);

#ifdef __cplusplus
}
#endif

#endif
