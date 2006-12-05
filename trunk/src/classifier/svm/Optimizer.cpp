/***********************************************************************/
/*                                                                     */
/*   Optimizer.cpp                                                     */
/*                                                                     */
/*   Interface to the PR_LOQO optimization package for SVM.            */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 19.07.99                                                    */
/*                                                                     */
/*   Copyright (c) 1999  Universitaet Dortmund - All rights reserved   */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include "classifier/svm/pr_loqo.h"
#include "classifier/svm/Optimizer.h"
#include "classifier/svm/SVM.h"

#ifdef USE_SVMLIGHT
#include "classifier/svm/SVM_light.h"
#endif

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

INT verbosity=1;

/* /////////////////////////////////////////////////////////////// */

double *optimize_qp();
double *primal=0,*dual=0;
double init_margin=0.15;
INT   init_iter=500,precision_violations=0;
double model_b;
double opt_precision=DEF_PRECISION;

/* /////////////////////////////////////////////////////////////// */

/* start the optimizer and return the optimal values */
double *optimize_qp(QP *qp,double *epsilon_crit, INT nx,double *threshold, INT& svm_maxqpsize)
{
  register INT i,j,result;
  double margin,obj_before,obj_after;
  double sigdig,dist,epsilon_loqo;
  int iter;
 
  if(!primal) { /* allocate memory at first call */
    primal=new double[nx*3];
    dual=new double[nx*2+1];
  }
  
  if(verbosity>=4) { /* really verbose */
    printf("\n\n");
    for(i=0;i<qp->opt_n;i++) {
      printf("%f: ",qp->opt_g0[i]);
      for(j=0;j<qp->opt_n;j++) {
	printf("%f ",qp->opt_g[i*qp->opt_n+j]);
      }
      printf(": a%d=%.10f < %f",i,qp->opt_xinit[i],qp->opt_up[i]);
      printf(": y=%f\n",qp->opt_ce[i]);
    }
    for(j=0;j<qp->opt_m;j++) {
      printf("EQ-%d: %f*a0",j,qp->opt_ce[j]);
      for(i=1;i<qp->opt_n;i++) {
	printf(" + %f*a%d",qp->opt_ce[i],i);
      }
      printf(" = %f\n\n",-qp->opt_ce0[0]);
    }
}

  obj_before=0; /* calculate objective before optimization */
  for(i=0;i<qp->opt_n;i++) {
    obj_before+=(qp->opt_g0[i]*qp->opt_xinit[i]);
    obj_before+=(0.5*qp->opt_xinit[i]*qp->opt_xinit[i]*qp->opt_g[i*qp->opt_n+i]);
    for(j=0;j<i;j++) {
      obj_before+=(qp->opt_xinit[j]*qp->opt_xinit[i]*qp->opt_g[j*qp->opt_n+i]);
    }
  }

  result=STILL_RUNNING;
  qp->opt_ce0[0]*=(-1.0);
  /* Run pr_loqo. If a run fails, try again with parameters which lead */
  /* to a slower, but more robust setting. */
  for(margin=init_margin,iter=init_iter;
      (margin<=0.9999999) && (result!=OPTIMAL_SOLUTION);) {

	opt_precision=CMath::max(opt_precision, DEF_PRECISION);
    sigdig=-log10(opt_precision);

    result=pr_loqo((int)qp->opt_n,(int)qp->opt_m,
		   (double *)qp->opt_g0,(double *)qp->opt_g,
		   (double *)qp->opt_ce,(double *)qp->opt_ce0,
		   (double *)qp->opt_low,(double *)qp->opt_up,
		   (double *)primal,(double *)dual, 
		   (int)(verbosity-2),
		   (double)sigdig,(int)iter, 
		   (double)margin,(double)(qp->opt_up[0])/4.0,(int)0);

    if(isnan(dual[0])) {     /* check for choldc problem */
      if(verbosity>=2) {
		  CIO::message(M_DEBUG, "Restarting PR_LOQO with more conservative parameters.\n");
      }
      if(init_margin<0.80) { /* become more conservative in general */
	init_margin=(4.0*margin+1.0)/5.0;
      }
      margin=(margin+1.0)/2.0;
      (opt_precision)*=10.0;   /* reduce precision */
      if(verbosity>=2) {
		  CIO::message(M_DEBUG, "Reducing precision of PR_LOQO.\n");
      }
    }
    else if(result!=OPTIMAL_SOLUTION) {
      iter+=2000; 
      init_iter+=10;
      (opt_precision)*=10.0;   /* reduce precision */
      if(verbosity>=2) {
		  CIO::message(M_DEBUG, "Reducing precision of PR_LOQO due to (%ld).\n",result);
      }      
    }
  }

  if(qp->opt_m)         /* Thanks to Alex Smola for this hint */
    model_b=dual[0];
  else
    model_b=0;

  /* Check the precision of the alphas. If results of current optimization */
  /* violate KT-Conditions, relax the epsilon on the bounds on alphas. */
  epsilon_loqo=1E-10;
  for(i=0;i<qp->opt_n;i++) {
    dist=-model_b*qp->opt_ce[i]; 
    dist+=(qp->opt_g0[i]+1.0);
    for(j=0;j<i;j++) {
      dist+=(primal[j]*qp->opt_g[j*qp->opt_n+i]);
    }
    for(j=i;j<qp->opt_n;j++) {
      dist+=(primal[j]*qp->opt_g[i*qp->opt_n+j]);
    }
    /*  printf("LOQO: a[%d]=%f, dist=%f, b=%f\n",i,primal[i],dist,dual[0]); */
    if((primal[i]<(qp->opt_up[i]-epsilon_loqo)) && (dist < (1.0-(*epsilon_crit)))) {
      epsilon_loqo=(qp->opt_up[i]-primal[i])*2.0;
    }
    else if((primal[i]>(0+epsilon_loqo)) && (dist > (1.0+(*epsilon_crit)))) {
      epsilon_loqo=primal[i]*2.0;
    }
  }

  for(i=0;i<qp->opt_n;i++) {  /* clip alphas to bounds */
    if(primal[i]<=(0+epsilon_loqo)) {
      primal[i]=0;
    }
    else if(primal[i]>=(qp->opt_up[i]-epsilon_loqo)) {
      primal[i]=qp->opt_up[i];
    }
  }

  obj_after=0;  /* calculate objective after optimization */
  for(i=0;i<qp->opt_n;i++) {
    obj_after+=(qp->opt_g0[i]*primal[i]);
    obj_after+=(0.5*primal[i]*primal[i]*qp->opt_g[i*qp->opt_n+i]);
    for(j=0;j<i;j++) {
      obj_after+=(primal[j]*primal[i]*qp->opt_g[j*qp->opt_n+i]);
    }
  }

  /* if optimizer returned NAN values, reset and retry with smaller */
  /* working set. */
  if(isnan(obj_after) || isnan(model_b)) {
    for(i=0;i<qp->opt_n;i++) {
      primal[i]=qp->opt_xinit[i];
    }     
    model_b=0;
    if(svm_maxqpsize>2) {
      svm_maxqpsize--;  /* decrease size of qp-subproblems */
    }
  }

  if(obj_after >= obj_before) { /* check whether there was progress */
    (opt_precision)/=100.0;
    precision_violations++;
    if(verbosity>=2) {
		CIO::message(M_DEBUG, "Increasing Precision of PR_LOQO.\n");
    }
  }

  if(precision_violations > 500) { 
    (*epsilon_crit)*=10.0;
    precision_violations=0;
	CIO::message(M_INFO, "Relaxing epsilon on KT-Conditions.\n");
  }	  

  (*threshold)=model_b;

  if(result!=OPTIMAL_SOLUTION) {
#ifdef HAVE_PYTHON
     throw SVMException("PR_LOQO did not converge.\n");
#else
	  CIO::message(M_ERROR, "PR_LOQO did not converge.\n");
#endif
    return(qp->opt_xinit);
  }
  else {
    return(primal);
  }
}

