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
#include "spdiag.h"
#include "libmatlbm.h"
#include "libmmfile.h"

static mxChar _array1_[130] = { 'R', 'u', 'n', '-', 't', 'i', 'm', 'e', ' ',
                                'E', 'r', 'r', 'o', 'r', ':', ' ', 'F', 'i',
                                'l', 'e', ':', ' ', 's', 'p', 'd', 'i', 'a',
                                'g', ' ', 'L', 'i', 'n', 'e', ':', ' ', '1',
                                ' ', 'C', 'o', 'l', 'u', 'm', 'n', ':', ' ',
                                '1', ' ', 'T', 'h', 'e', ' ', 'f', 'u', 'n',
                                'c', 't', 'i', 'o', 'n', ' ', '"', 's', 'p',
                                'd', 'i', 'a', 'g', '"', ' ', 'w', 'a', 's',
                                ' ', 'c', 'a', 'l', 'l', 'e', 'd', ' ', 'w',
                                'i', 't', 'h', ' ', 'm', 'o', 'r', 'e', ' ',
                                't', 'h', 'a', 'n', ' ', 't', 'h', 'e', ' ',
                                'd', 'e', 'c', 'l', 'a', 'r', 'e', 'd', ' ',
                                'n', 'u', 'm', 'b', 'e', 'r', ' ', 'o', 'f',
                                ' ', 'o', 'u', 't', 'p', 'u', 't', 's', ' ',
                                '(', '1', ')', '.' };
static mxArray * _mxarray0_;

static mxChar _array3_[129] = { 'R', 'u', 'n', '-', 't', 'i', 'm', 'e', ' ',
                                'E', 'r', 'r', 'o', 'r', ':', ' ', 'F', 'i',
                                'l', 'e', ':', ' ', 's', 'p', 'd', 'i', 'a',
                                'g', ' ', 'L', 'i', 'n', 'e', ':', ' ', '1',
                                ' ', 'C', 'o', 'l', 'u', 'm', 'n', ':', ' ',
                                '1', ' ', 'T', 'h', 'e', ' ', 'f', 'u', 'n',
                                'c', 't', 'i', 'o', 'n', ' ', '"', 's', 'p',
                                'd', 'i', 'a', 'g', '"', ' ', 'w', 'a', 's',
                                ' ', 'c', 'a', 'l', 'l', 'e', 'd', ' ', 'w',
                                'i', 't', 'h', ' ', 'm', 'o', 'r', 'e', ' ',
                                't', 'h', 'a', 'n', ' ', 't', 'h', 'e', ' ',
                                'd', 'e', 'c', 'l', 'a', 'r', 'e', 'd', ' ',
                                'n', 'u', 'm', 'b', 'e', 'r', ' ', 'o', 'f',
                                ' ', 'i', 'n', 'p', 'u', 't', 's', ' ', '(',
                                '1', ')', '.' };
static mxArray * _mxarray2_;
static mxArray * _mxarray4_;
static mxArray * _mxarray5_;

void InitializeModule_spdiag(void) {
    _mxarray0_ = mclInitializeString(130, _array1_);
    _mxarray2_ = mclInitializeString(129, _array3_);
    _mxarray4_ = mclInitializeDouble(1.0);
    _mxarray5_ = mclInitializeDouble(0.0);
}

void TerminateModule_spdiag(void) {
    mxDestroyArray(_mxarray5_);
    mxDestroyArray(_mxarray4_);
    mxDestroyArray(_mxarray2_);
    mxDestroyArray(_mxarray0_);
}

static mxArray * Mspdiag(int nargout_, mxArray * diagonal);

_mexLocalFunctionTable _local_function_table_spdiag
  = { 0, (mexFunctionTableEntry *)NULL };

/*
 * The function "mlfSpdiag" contains the normal interface for the "spdiag"
 * M-function from file "/opt/home/raetsch/matlab/utils/spdiag.m" (lines 1-9).
 * This function processes any input arguments and passes them to the
 * implementation version of the function, appearing above.
 */
mxArray * mlfSpdiag(mxArray * diagonal) {
    int nargout = 1;
    mxArray * A = mclGetUninitializedArray();
    mlfEnterNewContext(0, 1, diagonal);
    A = Mspdiag(nargout, diagonal);
    mlfRestorePreviousContext(0, 1, diagonal);
    return mlfReturnValue(A);
}

/*
 * The function "mlxSpdiag" contains the feval interface for the "spdiag"
 * M-function from file "/opt/home/raetsch/matlab/utils/spdiag.m" (lines 1-9).
 * The feval function calls the implementation version of spdiag through this
 * function. This function processes any input arguments and passes them to the
 * implementation version of the function, appearing above.
 */
void mlxSpdiag(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]) {
    mxArray * mprhs[1];
    mxArray * mplhs[1];
    int i;
    if (nlhs > 1) {
        mlfError(_mxarray0_);
    }
    if (nrhs > 1) {
        mlfError(_mxarray2_);
    }
    for (i = 0; i < 1; ++i) {
        mplhs[i] = mclGetUninitializedArray();
    }
    for (i = 0; i < 1 && i < nrhs; ++i) {
        mprhs[i] = prhs[i];
    }
    for (; i < 1; ++i) {
        mprhs[i] = NULL;
    }
    mlfEnterNewContext(0, 1, mprhs[0]);
    mplhs[0] = Mspdiag(nlhs, mprhs[0]);
    mlfRestorePreviousContext(0, 1, mprhs[0]);
    plhs[0] = mplhs[0];
}

/*
 * The function "Mspdiag" is the implementation version of the "spdiag"
 * M-function from file "/opt/home/raetsch/matlab/utils/spdiag.m" (lines 1-9).
 * It contains the actual compiled code for that M-function. It is a static
 * function and must only be called from one of the interface functions,
 * appearing below.
 */
/*
 * function A=spdiag(diagonal) ;
 */
static mxArray * Mspdiag(int nargout_, mxArray * diagonal) {
    mexLocalFunctionTable save_local_function_table_ = mclSetCurrentLocalFunctionTable(
                                                         &_local_function_table_spdiag);
    mxArray * A = mclGetUninitializedArray();
    mclCopyArray(&diagonal);
    /*
     * % A=spdiag(diagonal) ;
     * 
     * if size(diagonal,1)>1
     */
    if (mclGtBool(
          mclVe(
            mlfSize(
              mclValueVarargout(), mclVa(diagonal, "diagonal"), _mxarray4_)),
          _mxarray4_)) {
        /*
         * A=spdiags(diagonal, 0, length(diagonal),length(diagonal)) ;
         */
        mlfAssign(
          &A,
          mlfSpdiags(
            NULL,
            mclVa(diagonal, "diagonal"),
            _mxarray5_,
            mlfScalar(mclLengthInt(mclVa(diagonal, "diagonal"))),
            mlfScalar(mclLengthInt(mclVa(diagonal, "diagonal")))));
    /*
     * else
     */
    } else {
        /*
         * A=spdiags(diagonal', 0, length(diagonal),length(diagonal)) ;
         */
        mlfAssign(
          &A,
          mlfSpdiags(
            NULL,
            mlfCtranspose(mclVa(diagonal, "diagonal")),
            _mxarray5_,
            mlfScalar(mclLengthInt(mclVa(diagonal, "diagonal"))),
            mlfScalar(mclLengthInt(mclVa(diagonal, "diagonal")))));
    /*
     * end ;
     */
    }
    mclValidateOutput(A, 1, nargout_, "A", "spdiag");
    mxDestroyArray(diagonal);
    mclSetCurrentLocalFunctionTable(save_local_function_table_);
    return A;
}
