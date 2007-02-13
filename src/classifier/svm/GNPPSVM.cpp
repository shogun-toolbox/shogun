/*
 svm2_mex.c: MEX-file for binary SVM with L2-soft margin solver.

 Compile: 
  mex svm2_mex.c gnppsolver.c kernel_fun.c

 Synopsis:
  [Alpha,bias,exitflag,kercnt,access,trnerr,t,UB,LB,History] = 
     svm2_mex(data,labels,ker,arg,C,solver,tmax,tolabs,tolrel,thlb,cache,verb)

 Input:
  data [dim x num_data] Training vectors.
  labels [1 x num_data] Labels.
  ker [string] Kernel identifier.
  arg [1 x nargs] Kernel argument.
  C [1x1] Regularization constant.
  solver [string] Solver; options are 'mdm'.
  tmax [1x1] Maximal number of iterations.
  tolabs [1x1] Absolute tolerance stopping condition.
  tolrel [1x1] Relaitve tolerance stopping condition.
  thlb [1x1] Threshold on lower bound.
  cache [1x1] Number of columns of kernel matrix to be cached.
    It takes cache*num_data*size(double) bytes of memory.
  verb [1x1] If 1 then some info about the training is printed.

 Output:
  Alpha [nclass x num_data] Weights.
  bias [1x1] Bias.
  exitflag [1x1] Indicates which stopping condition was used:
    UB-LB <= tolabs           ->  exit_flag = 1   Abs. tolerance.
    (UB-LB)/(LB+1) <= tolrel  ->  exit_flag = 2   Relative tolerance.
    t >= tmax                 ->  exit_flag = 0   Number of iterations.
  kercnt [1x1] Number of kernel evaluations.
  access [1x1] Number or requested columns of the kernel matrix.
  trnerr [1x1] Training error.
  t [1x1] Number of iterations.
  UB [1x1] Upper bound on the optimal solution.
  LB [1x1] Lower bound on the optimal solution.
  History [2x(t+1)] UB and LB with respect to number of iterations.


 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Vojtech Franc
 */

#include "lib/io.h"
#include "classifier/svm/GNPPSVM.h"
#include "classifier/svm/gnpplib.h"

#define INDEX(ROW,COL,DIM) (((COL)*(DIM))+(ROW)) 

CGNPPSVM::CGNPPSVM() : CSVM()
{
}

CGNPPSVM::CGNPPSVM(DREAL C, CKernel* k, CLabels* lab)
{
	set_C(C,C);
	set_labels(lab);
	set_kernel(k);
}

CGNPPSVM::~CGNPPSVM()
{

}

bool CGNPPSVM::train()
{

	ASSERT(get_labels() && get_labels()->get_num_labels());
	INT num_data = get_labels()->get_num_labels();
	SG_INFO( "%d trainlabels\n", num_data);

	DREAL* vector_y = new double[num_data];
	ASSERT(vector_y);


	for (int i=0; i<num_data; i++)
		vector_y[i]=get_labels()->get_label(i);

	ASSERT(get_kernel());

    DREAL C = get_C1();
    INT tmax = 1000000000;
    DREAL tolabs = 0;
    DREAL tolrel = epsilon;

    DREAL reg_const=0;
    if( C!=0 )
      reg_const = 1/C; 
   

    DREAL* diagK = new DREAL[num_data];
    ASSERT(diagK);

    for(INT i = 0; i < num_data; i++ ) {
      diagK[i] = 2*get_kernel()->kernel(i,i) + reg_const;
    }

    DREAL* alpha = new DREAL[num_data];
    ASSERT(alpha);
    DREAL* vector_c = new DREAL[num_data];
    ASSERT(vector_c);

    memset(vector_c,0,num_data*sizeof(DREAL));

    DREAL thlb = 0;
    INT t = 0;
    DREAL* History = NULL;
    INT verb = 0;
    DREAL aHa11, aHa22;

    CGNPPLib npp(vector_y,get_kernel(),num_data, reg_const);

    npp.gnpp_imdm(diagK, vector_c, vector_y, num_data, 
         tmax, tolabs, tolrel, thlb, alpha, &t, &aHa11, &aHa22, 
         &History, verb ); 

    INT num_sv = 0;
    DREAL nconst = History[INDEX(1,t,2)];
    DREAL trnerr = 0; /* counter of training error */

    for(INT i = 0; i < num_data; i++ )
    {
      if( alpha[i] != 0 ) num_sv++;
      if(vector_y[i] == 1) 
      {
        alpha[i] = alpha[i]*2/nconst;
        if( alpha[i]/(2*C) >= 1 ) trnerr++;
      }
      else
      {
        alpha[i] = -alpha[i]*2/nconst;
        if( alpha[i]/(2*C) <= -1 ) trnerr++;
      }
    }
  
    DREAL b = 0.5*(aHa22 - aHa11)/nconst;;

	create_new_model(num_sv);
	CSVM::set_objective(nconst);

    set_bias(b);
    INT j = 0;
	for (int i=0; i<num_data; i++)
    {
      if( alpha[i] !=0)
      {
        set_support_vector(j, i);
		set_alpha(j, alpha[i]);
      j++;
    }
    }

return true;


}

