/*
 * MATLAB Compiler: 2.1
 * Date: Mon Jan 28 09:59:03 2002
 * Arguments: "-B" "macro_default" "-O" "all" "-O" "fold_scalar_mxarrays:on"
 * "-O" "fold_non_scalar_mxarrays:on" "-O" "optimize_integer_for_loops:on" "-O"
 * "array_indexing:on" "-O" "optimize_conditionals:on" "-m" "-W" "main" "-L"
 * "C" "-t" "-T" "link:exe" "-h" "libmmfile.mlib" "cleaner" 
 */

#ifndef MLF_V2
#define MLF_V2 1
#endif

#include "libmatlb.h"
#include "cleaner.h"
#include "libmmfile.h"

static mexFunctionTableEntry function_table[1]
  = { { "cleaner", mlxCleaner, 2, 1, &_local_function_table_cleaner } };

static _mexInitTermTableEntry init_term_table[2]
  = { { libmmfileInitialize, libmmfileTerminate },
      { InitializeModule_cleaner, TerminateModule_cleaner } };

static _mex_information _main_info
  = { 1, 1, function_table, 0, NULL, 0, NULL, 2, init_term_table };

/*
 * The function "main" is a Compiler-generated main wrapper, suitable for
 * building a stand-alone application.  It calls a library function to perform
 * initialization, call the main function, and perform library termination.
 */
//int main(int argc, const char * * argv) {
//    return mclMain(argc, argv, mlxCleaner, 1, &_main_info);
//}

void cleaner_main(double *covZ, int dim, double thresh,
		  double **T, int *num_dim) 
{
  mxArray *mxcovZ, *mxthresh, *mxT ;
  int i ;

  /*  mclInitMatlabRoot( 0, NULL );*/

  /*  mclLibInitGlobals(1, global_table) ;*/
  mlfFunctionTableSetup(1, function_table);
  mclLibInitInitTerms(2, init_term_table) ;
  /*mclAddPaths( 2, path_list_);*/

  if (dim>0)
    {
      mxthresh=mclInitializeDouble(thresh) ;
      mxcovZ=mclInitializeDoubleVector(dim, dim, covZ) ;

      mxT=mlfCleaner(mxcovZ, mxthresh) ;
      *num_dim=mxGetM(mxT) ;
      *T=(double*)malloc(sizeof(double)*(*num_dim)*dim) ;
      
      for (i=0; i<dim*(*num_dim); i++)
	(*T)[i]=mxGetPr(mxT)[i] ;

      mxDestroyArray(mxthresh);
      mxDestroyArray(mxcovZ);
      mxDestroyArray(mxT);
    } ;

  /*mclLibTermGlobals(1, global_table) ;*/
  mlfFunctionTableTakedown(1, function_table);
  mclLibTermInitTerms(2, init_term_table) ;
}

