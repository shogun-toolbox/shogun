#ifdef SVMCPLEX

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

#include "libmatlb.h"
#include "train_svm.h"
#include "cplex_init_mex_interface.h"
#include "qp_solve_mex_interface.h"
#include "spdiag.h"
#include "libmmfile.h"

mxArray * lpenv = NULL;

static mexGlobalTableEntry global_table[1] = { { "lpenv", &lpenv } };

static mexFunctionTableEntry function_table[4]
  = { { "train_svm", mlxTrain_svm, 3, 2, &_local_function_table_train_svm },
      { "cplex_init", mlxCplex_init, -1, -1,
        &_local_function_table_cplex_init },
      { "qp_solve", mlxQp_solve, -1, -1, &_local_function_table_qp_solve },
      { "spdiag", mlxSpdiag, 1, 1, &_local_function_table_spdiag } };

static const char * path_list_[2] = { "/opt/home/raetsch/CANDY/matlab/cplex", 
				      "/home/neuro/raetsch/CANDY/matlab/cplex" };

static _mexInitTermTableEntry init_term_table[5]
  = { { libmmfileInitialize, libmmfileTerminate },
      { InitializeModule_train_svm, TerminateModule_train_svm },
      { InitializeModule_cplex_init_mex_interface,
        TerminateModule_cplex_init_mex_interface },
      { InitializeModule_qp_solve_mex_interface,
        TerminateModule_qp_solve_mex_interface },
      { InitializeModule_spdiag, TerminateModule_spdiag } };

/*static _mex_information _main_info
  = { 1, 4, function_table, 1, global_table, 2,
  path_list_, 5, init_term_table };*/

/*
 * The function "main" is a Compiler-generated main wrapper, suitable for
 * building a stand-alone application.  It calls a library function to perform
 * initialization, call the main function, and perform library termination.
 */
void train_svm_main(double *XT, double *LT, double C, 
		    int ell, int dim, double *w, double *b) 
{
  mxArray *mxXT, *mxLT, *mxC, *mxw, *mxb ;
  int i ;

  /*  mclInitMatlabRoot( 0, NULL );*/

  mclLibInitGlobals(1, global_table) ;
  mlfFunctionTableSetup(5, function_table);
  mclLibInitInitTerms(6, init_term_table) ;
  mclAddPaths( 2, path_list_);

  if ((dim>0) & (ell>0))
    {
      mxC=mclInitializeDouble(C) ;
      mxXT=mclInitializeDoubleVector(dim, ell, XT) ;
      mxLT=mclInitializeDoubleVector(1, ell, LT) ;

      mxw=mlfTrain_svm(&mxb, mxXT, mxLT, mxC) ;
      
      *b=mxGetPr(mxb)[0] ;
      for (i=0; i<dim; i++)
	w[i]=mxGetPr(mxw)[i] ;

      mxDestroyArray(mxC);
      mxDestroyArray(mxXT);
      mxDestroyArray(mxLT);
      mxDestroyArray(mxb);
      mxDestroyArray(mxw);
    } ;

  mclLibTermGlobals(1, global_table) ;
  mlfFunctionTableTakedown(5, function_table);
  mclLibTermInitTerms(6, init_term_table) ;
}

#endif // SVMCPLEX

