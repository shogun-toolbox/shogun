#ifdef WITHMATLAB

/*
 * MATLAB Compiler: 2.1
 * Date: Mon Jan 28 12:05:31 2002
 * Arguments: "-B" "macro_default" "-O" "all" "-O" "fold_scalar_mxarrays:on"
 * "-O" "fold_non_scalar_mxarrays:on" "-O" "optimize_integer_for_loops:on" "-O"
 * "array_indexing:on" "-O" "optimize_conditionals:on" "-m" "-W" "main" "-L"
 * "C" "-t" "-T" "link:exe" "-h" "libmmfile.mlib" "cleaner" 
 */
#include "cleaner.h"
#include "libmatlbm.h"
#include "libmmfile.h"

static mxChar _array1_[132] = { 'R', 'u', 'n', '-', 't', 'i', 'm', 'e', ' ',
                                'E', 'r', 'r', 'o', 'r', ':', ' ', 'F', 'i',
                                'l', 'e', ':', ' ', 'c', 'l', 'e', 'a', 'n',
                                'e', 'r', ' ', 'L', 'i', 'n', 'e', ':', ' ',
                                '1', ' ', 'C', 'o', 'l', 'u', 'm', 'n', ':',
                                ' ', '1', ' ', 'T', 'h', 'e', ' ', 'f', 'u',
                                'n', 'c', 't', 'i', 'o', 'n', ' ', '"', 'c',
                                'l', 'e', 'a', 'n', 'e', 'r', '"', ' ', 'w',
                                'a', 's', ' ', 'c', 'a', 'l', 'l', 'e', 'd',
                                ' ', 'w', 'i', 't', 'h', ' ', 'm', 'o', 'r',
                                'e', ' ', 't', 'h', 'a', 'n', ' ', 't', 'h',
                                'e', ' ', 'd', 'e', 'c', 'l', 'a', 'r', 'e',
                                'd', ' ', 'n', 'u', 'm', 'b', 'e', 'r', ' ',
                                'o', 'f', ' ', 'o', 'u', 't', 'p', 'u', 't',
                                's', ' ', '(', '1', ')', '.' };
static mxArray * _mxarray0_;

static mxChar _array3_[131] = { 'R', 'u', 'n', '-', 't', 'i', 'm', 'e', ' ',
                                'E', 'r', 'r', 'o', 'r', ':', ' ', 'F', 'i',
                                'l', 'e', ':', ' ', 'c', 'l', 'e', 'a', 'n',
                                'e', 'r', ' ', 'L', 'i', 'n', 'e', ':', ' ',
                                '1', ' ', 'C', 'o', 'l', 'u', 'm', 'n', ':',
                                ' ', '1', ' ', 'T', 'h', 'e', ' ', 'f', 'u',
                                'n', 'c', 't', 'i', 'o', 'n', ' ', '"', 'c',
                                'l', 'e', 'a', 'n', 'e', 'r', '"', ' ', 'w',
                                'a', 's', ' ', 'c', 'a', 'l', 'l', 'e', 'd',
                                ' ', 'w', 'i', 't', 'h', ' ', 'm', 'o', 'r',
                                'e', ' ', 't', 'h', 'a', 'n', ' ', 't', 'h',
                                'e', ' ', 'd', 'e', 'c', 'l', 'a', 'r', 'e',
                                'd', ' ', 'n', 'u', 'm', 'b', 'e', 'r', ' ',
                                'o', 'f', ' ', 'i', 'n', 'p', 'u', 't', 's',
                                ' ', '(', '2', ')', '.' };
static mxArray * _mxarray2_;

static mxChar _array5_[32] = { 'c', 'o', 'm', 'p', 'u', 't', 'i', 'n', 'g',
                               ' ', 'P', 'C', 'A', ' ', 'i', 'n', ' ', '%',
                               'i', ' ', 'd', 'i', 'm', 'e', 'n', 's', 'i',
                               'o', 'n', 's', 0x005c, 'n' };
static mxArray * _mxarray4_;
static mxArray * _mxarray6_;

static mxChar _array8_[38] = { 'r', 'e', 'd', 'u', 'c', 'i', 'n', 'g', ' ', 't',
                               'o', ' ', '%', 'i', ' ', 'd', 'i', 'm', 'e', 'n',
                               's', 'i', 'o', 'n', 's', ' ', '(', 'E', 'V', ':',
                               '%', 'e', '-', '%', 'e', ')', 0x005c, 'n' };
static mxArray * _mxarray7_;
static mxArray * _mxarray9_;
static mxArray * _mxarray10_;

void InitializeModule_cleaner(void) {
    _mxarray0_ = mclInitializeString(132, _array1_);
    _mxarray2_ = mclInitializeString(131, _array3_);
    _mxarray4_ = mclInitializeString(32, _array5_);
    _mxarray6_ = mclInitializeDouble(1.0);
    _mxarray7_ = mclInitializeString(38, _array8_);
    _mxarray9_ = mclInitializeDouble(-.5);
    _mxarray10_ = mclInitializeDouble(0.0);
}

void TerminateModule_cleaner(void) {
    mxDestroyArray(_mxarray10_);
    mxDestroyArray(_mxarray9_);
    mxDestroyArray(_mxarray7_);
    mxDestroyArray(_mxarray6_);
    mxDestroyArray(_mxarray4_);
    mxDestroyArray(_mxarray2_);
    mxDestroyArray(_mxarray0_);
}

static mxArray * Mcleaner(INT nargout_, mxArray * covz, mxArray * thresh);

_mexLocalFunctionTable _local_function_table_cleaner
  = { 0, (mexFunctionTableEntry *)NULL };

/*
 * The function "mlfCleaner" contains the normal interface for the "cleaner"
 * M-function from file "/opt/home/raetsch/cleaner.m" (lines 1-26). This
 * function processes any input arguments and passes them to the implementation
 * version of the function, appearing above.
 */
mxArray * mlfCleaner(mxArray * covz, mxArray * thresh) {
    INT nargout = 1;
    mxArray * T = mclGetUninitializedArray();
    mlfEnterNewContext(0, 2, covz, thresh);
    T = Mcleaner(nargout, covz, thresh);
    mlfRestorePreviousContext(0, 2, covz, thresh);
    return mlfReturnValue(T);
}

/*
 * The function "mlxCleaner" contains the feval interface for the "cleaner"
 * M-function from file "/opt/home/raetsch/cleaner.m" (lines 1-26). The feval
 * function calls the implementation version of cleaner through this function.
 * This function processes any input arguments and passes them to the
 * implementation version of the function, appearing above.
 */
void mlxCleaner(INT nlhs, mxArray * plhs[], INT nrhs, mxArray * prhs[]) {
    mxArray * mprhs[2];
    mxArray * mplhs[1];
    INT i;
    if (nlhs > 1) {
        mlfError(_mxarray0_);
    }
    if (nrhs > 2) {
        mlfError(_mxarray2_);
    }
    for (i = 0; i < 1; ++i) {
        mplhs[i] = mclGetUninitializedArray();
    }
    for (i = 0; i < 2 && i < nrhs; ++i) {
        mprhs[i] = prhs[i];
    }
    for (; i < 2; ++i) {
        mprhs[i] = NULL;
    }
    mlfEnterNewContext(0, 2, mprhs[0], mprhs[1]);
    mplhs[0] = Mcleaner(nlhs, mprhs[0], mprhs[1]);
    mlfRestorePreviousContext(0, 2, mprhs[0], mprhs[1]);
    plhs[0] = mplhs[0];
}

/*
 * The function "Mcleaner" is the implementation version of the "cleaner"
 * M-function from file "/opt/home/raetsch/cleaner.m" (lines 1-26). It contains
 * the actual compiled code for that M-function. It is a static function and
 * must only be called from one of the interface functions, appearing below.
 */
/*
 * function T=cleaner(covz, thresh) ;
 */
static mxArray * Mcleaner(INT nargout_, mxArray * covz, mxArray * thresh) {
    mexLocalFunctionTable save_local_function_table_ = mclSetCurrentLocalFunctionTable(
                                                         &_local_function_table_cleaner);
    mxArray * T = mclGetUninitializedArray();
    mxArray * dinv = mclGetUninitializedArray();
    mxArray * dgood = mclGetUninitializedArray();
    mxArray * v = mclGetUninitializedArray();
    mxArray * d = mclGetUninitializedArray();
    mxArray * ans = mclGetUninitializedArray();
    mclCopyArray(&covz);
    mclCopyArray(&thresh);
    /*
     * 
     * fprintf('computing PCA in %i dimensions\n', size(covz,1)) ;
     */
    mclAssignAns(
      &ans,
      mlfNFprintf(
        0,
        _mxarray4_,
        mclVe(mlfSize(mclValueVarargout(), mclVa(covz, "covz"), _mxarray6_)),
        NULL));
    /*
     * 
     * %covz = Z * Z';
     * %covz = covz / size(Z,2);			% rescale with sample size
     * 
     * d = eig(covz)
     */
    mlfAssign(&d, mlfEig(NULL, mclVa(covz, "covz"), NULL, NULL));
    mclPrintArray(mclVsv(d, "d"), "d");
    /*
     * 
     * [v, d] = eig(covz);			% get the eigensystem,
     */
    mlfAssign(&v, mlfEig(&d, mclVa(covz, "covz"), NULL, NULL));
    mclPrintArray(mclVsv(v, "v"), "v");
    /*
     * % negative eigenvalues have
     * % to go ...
     * d = diag(d);				% cut out diagonal
     */
    mlfAssign(&d, mlfDiag(mclVv(d, "d"), NULL));
    /*
     * dgood = d > thresh ;
     */
    mlfAssign(&dgood, mclGt(mclVv(d, "d"), mclVa(thresh, "thresh")));
    /*
     * 
     * fprintf('reducing to %i dimensions (EV:%e-%e)\n', sum(dgood), min(d(dgood)), max(d(dgood))) ;
     */
    mclAssignAns(
      &ans,
      mlfNFprintf(
        0,
        _mxarray7_,
        mclVe(mlfSum(mclVv(dgood, "dgood"), NULL)),
        mclVe(
          mlfMin(
            NULL,
            mclVe(mclArrayRef1(mclVsv(d, "d"), mclVsv(dgood, "dgood"))),
            NULL,
            NULL)),
        mclVe(
          mlfMax(
            NULL,
            mclVe(mclArrayRef1(mclVsv(d, "d"), mclVsv(dgood, "dgood"))),
            NULL,
            NULL)),
        NULL));
    /*
     * dinv = spdiags(d(dgood).^(-1/2), 0, sum(dgood), sum(dgood));
     */
    mlfAssign(
      &dinv,
      mlfSpdiags(
        NULL,
        mlfPower(
          mclVe(mclArrayRef1(mclVsv(d, "d"), mclVsv(dgood, "dgood"))),
          _mxarray9_),
        _mxarray10_,
        mclVe(mlfSum(mclVv(dgood, "dgood"), NULL)),
        mclVe(mlfSum(mclVv(dgood, "dgood"), NULL))));
    /*
     * 
     * T=dinv*v(:,dgood)' ;
     */
    mlfAssign(
      &T,
      mclMtimes(
        mclVv(dinv, "dinv"),
        mlfCtranspose(
          mclVe(
            mclArrayRef2(
              mclVsv(v, "v"),
              mlfCreateColonIndex(),
              mclVsv(dgood, "dgood"))))));
    mclPrintArray(mclVsv(T, "T"), "T");
    mclValidateOutput(T, 1, nargout_, "T", "cleaner");
    mxDestroyArray(ans);
    mxDestroyArray(d);
    mxDestroyArray(v);
    mxDestroyArray(dgood);
    mxDestroyArray(dinv);
    mxDestroyArray(thresh);
    mxDestroyArray(covz);
    mclSetCurrentLocalFunctionTable(save_local_function_table_);
    return T;
    /*
     * 
     * %fprintf('done') ;
     * 
     * %zeigen = v' * Z;
     * %zeigen = dinv * zeigen(dgood,:);
     * 
     */
}

#endif
