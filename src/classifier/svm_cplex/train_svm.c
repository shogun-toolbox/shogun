#ifdef USE_SVMCPLEX

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
#include "train_svm.h"
#include "cplex_init_mex_interface.h"
#include "libmatlbm.h"
#include "qp_solve_mex_interface.h"
#include "spdiag.h"

extern mxArray * lpenv;

static mxChar _array1_[136] = { 'R', 'u', 'n', '-', 't', 'i', 'm', 'e', ' ',
                                'E', 'r', 'r', 'o', 'r', ':', ' ', 'F', 'i',
                                'l', 'e', ':', ' ', 't', 'r', 'a', 'i', 'n',
                                '_', 's', 'v', 'm', ' ', 'L', 'i', 'n', 'e',
                                ':', ' ', '1', ' ', 'C', 'o', 'l', 'u', 'm',
                                'n', ':', ' ', '1', ' ', 'T', 'h', 'e', ' ',
                                'f', 'u', 'n', 'c', 't', 'i', 'o', 'n', ' ',
                                '"', 't', 'r', 'a', 'i', 'n', '_', 's', 'v',
                                'm', '"', ' ', 'w', 'a', 's', ' ', 'c', 'a',
                                'l', 'l', 'e', 'd', ' ', 'w', 'i', 't', 'h',
                                ' ', 'm', 'o', 'r', 'e', ' ', 't', 'h', 'a',
                                'n', ' ', 't', 'h', 'e', ' ', 'd', 'e', 'c',
                                'l', 'a', 'r', 'e', 'd', ' ', 'n', 'u', 'm',
                                'b', 'e', 'r', ' ', 'o', 'f', ' ', 'o', 'u',
                                't', 'p', 'u', 't', 's', ' ', '(', '2', ')',
                                '.' };
static mxArray * _mxarray0_;

static mxChar _array3_[135] = { 'R', 'u', 'n', '-', 't', 'i', 'm', 'e', ' ',
                                'E', 'r', 'r', 'o', 'r', ':', ' ', 'F', 'i',
                                'l', 'e', ':', ' ', 't', 'r', 'a', 'i', 'n',
                                '_', 's', 'v', 'm', ' ', 'L', 'i', 'n', 'e',
                                ':', ' ', '1', ' ', 'C', 'o', 'l', 'u', 'm',
                                'n', ':', ' ', '1', ' ', 'T', 'h', 'e', ' ',
                                'f', 'u', 'n', 'c', 't', 'i', 'o', 'n', ' ',
                                '"', 't', 'r', 'a', 'i', 'n', '_', 's', 'v',
                                'm', '"', ' ', 'w', 'a', 's', ' ', 'c', 'a',
                                'l', 'l', 'e', 'd', ' ', 'w', 'i', 't', 'h',
                                ' ', 'm', 'o', 'r', 'e', ' ', 't', 'h', 'a',
                                'n', ' ', 't', 'h', 'e', ' ', 'd', 'e', 'c',
                                'l', 'a', 'r', 'e', 'd', ' ', 'n', 'u', 'm',
                                'b', 'e', 'r', ' ', 'o', 'f', ' ', 'i', 'n',
                                'p', 'u', 't', 's', ' ', '(', '3', ')', '.' };
static mxArray * _mxarray2_;
static mxArray * _mxarray4_;
static mxArray * _mxarray5_;
static mxArray * _mxarray6_;
static mxArray * _mxarray7_;
static mxArray * _mxarray8_;

void InitializeModule_train_svm(void) {
    _mxarray0_ = mclInitializeString(136, _array1_);
    _mxarray2_ = mclInitializeString(135, _array3_);
    _mxarray4_ = mclInitializeDouble(0.0);
    _mxarray5_ = mclInitializeDouble(1.0);
    _mxarray6_ = mclInitializeDouble(1e+20);
    _mxarray7_ = mclInitializeDoubleVector(0, 0, (double *)NULL);
    _mxarray8_ = mclInitializeDouble(2.0);
}

void TerminateModule_train_svm(void) {
    mxDestroyArray(_mxarray8_);
    mxDestroyArray(_mxarray7_);
    mxDestroyArray(_mxarray6_);
    mxDestroyArray(_mxarray5_);
    mxDestroyArray(_mxarray4_);
    mxDestroyArray(_mxarray2_);
    mxDestroyArray(_mxarray0_);
}

static mxArray * Mtrain_svm(mxArray * * b,
                            int nargout_,
                            mxArray * XT,
                            mxArray * LT,
                            mxArray * C);

_mexLocalFunctionTable _local_function_table_train_svm
  = { 0, (mexFunctionTableEntry *)NULL };

/*
 * The function "mlfTrain_svm" contains the normal interface for the
 * "train_svm" M-function from file
 * "/opt/home/raetsch/cvs.II/Genefinder.cvs8/src/svm_cplex/train_svm.m" (lines
 * 1-35). This function processes any input arguments and passes them to the
 * implementation version of the function, appearing above.
 */
mxArray * mlfTrain_svm(mxArray * * b, mxArray * XT, mxArray * LT, mxArray * C) {
    int nargout = 1;
    mxArray * w = mclGetUninitializedArray();
    mxArray * b__ = mclGetUninitializedArray();
    mlfEnterNewContext(1, 3, b, XT, LT, C);
    if (b != NULL) {
        ++nargout;
    }
    w = Mtrain_svm(&b__, nargout, XT, LT, C);
    mlfRestorePreviousContext(1, 3, b, XT, LT, C);
    if (b != NULL) {
        mclCopyOutputArg(b, b__);
    } else {
        mxDestroyArray(b__);
    }
    return mlfReturnValue(w);
}

/*
 * The function "mlxTrain_svm" contains the feval interface for the "train_svm"
 * M-function from file
 * "/opt/home/raetsch/cvs.II/Genefinder.cvs8/src/svm_cplex/train_svm.m" (lines
 * 1-35). The feval function calls the implementation version of train_svm
 * through this function. This function processes any input arguments and
 * passes them to the implementation version of the function, appearing above.
 */
void mlxTrain_svm(int nlhs, mxArray * plhs[], int nrhs, mxArray * prhs[]) {
    mxArray * mprhs[3];
    mxArray * mplhs[2];
    int i;
    if (nlhs > 2) {
        mlfError(_mxarray0_);
    }
    if (nrhs > 3) {
        mlfError(_mxarray2_);
    }
    for (i = 0; i < 2; ++i) {
        mplhs[i] = mclGetUninitializedArray();
    }
    for (i = 0; i < 3 && i < nrhs; ++i) {
        mprhs[i] = prhs[i];
    }
    for (; i < 3; ++i) {
        mprhs[i] = NULL;
    }
    mlfEnterNewContext(0, 3, mprhs[0], mprhs[1], mprhs[2]);
    mplhs[0] = Mtrain_svm(&mplhs[1], nlhs, mprhs[0], mprhs[1], mprhs[2]);
    mlfRestorePreviousContext(0, 3, mprhs[0], mprhs[1], mprhs[2]);
    plhs[0] = mplhs[0];
    for (i = 1; i < 2 && i < nlhs; ++i) {
        plhs[i] = mplhs[i];
    }
    for (; i < 2; ++i) {
        mxDestroyArray(mplhs[i]);
    }
}

/*
 * The function "Mtrain_svm" is the implementation version of the "train_svm"
 * M-function from file
 * "/opt/home/raetsch/cvs.II/Genefinder.cvs8/src/svm_cplex/train_svm.m" (lines
 * 1-35). It contains the actual compiled code for that M-function. It is a
 * static function and must only be called from one of the interface functions,
 * appearing below.
 */
/*
 * function [w,b]=train_svm(XT, LT, C)
 */
static mxArray * Mtrain_svm(mxArray * * b,
                            int nargout_,
                            mxArray * XT,
                            mxArray * LT,
                            mxArray * C) {
    mexLocalFunctionTable save_local_function_table_ = mclSetCurrentLocalFunctionTable(
                                                         &_local_function_table_train_svm);
    mxArray * w = mclGetUninitializedArray();
    mxArray * lambda = mclGetUninitializedArray();
    mxArray * res = mclGetUninitializedArray();
    mxArray * A = mclGetUninitializedArray();
    mxArray * f = mclGetUninitializedArray();
    mxArray * i = mclGetUninitializedArray();
    mxArray * Q = mclGetUninitializedArray();
    mxArray * UB = mclGetUninitializedArray();
    mxArray * LB = mclGetUninitializedArray();
    mxArray * INF = mclGetUninitializedArray();
    mxArray * ell = mclGetUninitializedArray();
    mxArray * dim = mclGetUninitializedArray();
    mxArray * ans = mclGetUninitializedArray();
    mclCopyArray(&XT);
    mclCopyArray(&LT);
    mclCopyArray(&C);
    /*
     * % [w,b]=train_svm(XT, LT, C)
     * 
     * global lpenv ;
     * 
     * [dim,ell]=size(XT) ;
     */
    mlfSize(mlfVarargout(&dim, &ell, NULL), mclVa(XT, "XT"), NULL);
    /*
     * 
     * if isempty(lpenv),
     */
    if (mlfTobool(mclVe(mlfIsempty(mclVg(&lpenv, "lpenv"))))) {
        /*
         * lpenv=0 ;
         */
        mlfAssign(mclPrepareGlobal(&lpenv), _mxarray4_);
    /*
     * end ;
     */
    }
    /*
     * if lpenv==0,
     */
    if (mclEqBool(mclVg(&lpenv, "lpenv"), _mxarray4_)) {
        /*
         * lpenv=cplex_init(1) ;
         */
        mlfAssign(
          mclPrepareGlobal(&lpenv),
          mlfNCplex_init(0, mclValueVarargout(), _mxarray5_, NULL));
    /*
     * end ;
     */
    }
    /*
     * 
     * INF=1e20 ;
     */
    mlfAssign(&INF, _mxarray6_);
    /*
     * %      b        w,                xi
     * LB= [ -INF;  -INF*ones(dim,1); zeros(ell,1)] ;
     */
    mlfAssign(
      &LB,
      mlfVertcat(
        mclUminus(mclVv(INF, "INF")),
        mclMtimes(
          mclUminus(mclVv(INF, "INF")),
          mclVe(mlfOnes(mclVv(dim, "dim"), _mxarray5_, NULL))),
        mclVe(mlfZeros(mclVv(ell, "ell"), _mxarray5_, NULL)),
        NULL));
    /*
     * UB= [ INF;    INF*ones(dim,1); INF*ones(ell,1)] ;
     */
    mlfAssign(
      &UB,
      mlfVertcat(
        mclVv(INF, "INF"),
        mclMtimes(
          mclVv(INF, "INF"),
          mclVe(mlfOnes(mclVv(dim, "dim"), _mxarray5_, NULL))),
        mclMtimes(
          mclVv(INF, "INF"),
          mclVe(mlfOnes(mclVv(ell, "ell"), _mxarray5_, NULL))),
        NULL));
    /*
     * Q=sparse(0) ;
     */
    mlfAssign(&Q, mlfSparse(_mxarray4_, NULL, NULL, NULL, NULL, NULL));
    /*
     * Q(dim+ell+1,dim+ell+1)=0 ;
     */
    mclArrayAssign2(
      &Q,
      _mxarray4_,
      mclPlus(mclPlus(mclVv(dim, "dim"), mclVv(ell, "ell")), _mxarray5_),
      mclPlus(mclPlus(mclVv(dim, "dim"), mclVv(ell, "ell")), _mxarray5_));
    /*
     * for i=1:dim, Q(1+i,1+i)=1; end;
     */
    {
        int v_ = mclForIntStart(1);
        int e_ = mclForIntEnd(mclVv(dim, "dim"));
        if (v_ > e_) {
            mlfAssign(&i, _mxarray7_);
        } else {
            for (; ; ) {
                mclIntArrayAssign2(&Q, _mxarray5_, 1 + v_, 1 + v_);
                if (v_ == e_) {
                    break;
                }
                ++v_;
            }
            mlfAssign(&i, mlfScalar(v_));
        }
    }
    /*
     * %Q(2:1+dim,2:1+dim)=speye(dim) ;
     * 
     * f=[zeros(dim+1,1); C*ones(ell,1)] ;
     */
    mlfAssign(
      &f,
      mlfVertcat(
        mclVe(
          mlfZeros(mclPlus(mclVv(dim, "dim"), _mxarray5_), _mxarray5_, NULL)),
        mclMtimes(
          mclVa(C, "C"), mclVe(mlfOnes(mclVv(ell, "ell"), _mxarray5_, NULL))),
        NULL));
    /*
     * A=sparse([-LT' -spdiag(LT)*XT' -eye(ell)]) ;
     */
    mlfAssign(
      &A,
      mlfSparse(
        mlfHorzcat(
          mclUminus(mlfCtranspose(mclVa(LT, "LT"))),
          mclMtimes(
            mclUminus(mclVe(mlfSpdiag(mclVa(LT, "LT")))),
            mlfCtranspose(mclVa(XT, "XT"))),
          mclUminus(mclVe(mlfEye(mclVv(ell, "ell"), NULL))),
          NULL),
        NULL,
        NULL,
        NULL,
        NULL,
        NULL));
    /*
     * b=-ones(ell,1) ;
     */
    mlfAssign(
      b, mclUminus(mclVe(mlfOnes(mclVv(ell, "ell"), _mxarray5_, NULL))));
    /*
     * [res,lambda]=qp_solve(lpenv, Q, f, A, b, LB, UB, 0, 1) ;
     */
    mlfNQp_solve(
      0,
      mlfVarargout(&res, &lambda, NULL),
      mclVg(&lpenv, "lpenv"),
      mclVv(Q, "Q"),
      mclVv(f, "f"),
      mclVv(A, "A"),
      mclVv(*b, "b"),
      mclVv(LB, "LB"),
      mclVv(UB, "UB"),
      _mxarray4_,
      _mxarray5_,
      NULL);
    /*
     * 
     * b=res(1) ;
     */
    mlfAssign(b, mclIntArrayRef1(mclVsv(res, "res"), 1));
    /*
     * w=res(2:dim+1)' ;
     */
    mlfAssign(
      &w,
      mlfCtranspose(
        mclVe(
          mclArrayRef1(
            mclVsv(res, "res"),
            mlfColon(
              _mxarray8_, mclPlus(mclVv(dim, "dim"), _mxarray5_), NULL)))));
    mclValidateOutput(w, 1, nargout_, "w", "train_svm");
    mclValidateOutput(*b, 2, nargout_, "b", "train_svm");
    mxDestroyArray(ans);
    mxDestroyArray(dim);
    mxDestroyArray(ell);
    mxDestroyArray(INF);
    mxDestroyArray(LB);
    mxDestroyArray(UB);
    mxDestroyArray(Q);
    mxDestroyArray(i);
    mxDestroyArray(f);
    mxDestroyArray(A);
    mxDestroyArray(res);
    mxDestroyArray(lambda);
    mxDestroyArray(C);
    mxDestroyArray(LT);
    mxDestroyArray(XT);
    mclSetCurrentLocalFunctionTable(save_local_function_table_);
    return w;
    /*
     * 
     * %out=w*XT+b
     * 
     * %cplex_quit(lpenv,1)
     */
}

#endif // USE_SVMCPLEX
