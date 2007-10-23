/*--------------------------------------------------------------------
 *  Implementation of SVM-Ocas solver. 
 *
 *  Linear unbiased binary SVM solver.
 *
 * Modifications:
 * 23-oct-2007, VF
 * 10-oct-2007, VF, created.
 * ----------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>

#include "classifier/svm/libocas.h"
#include "classifier/svm/libocas_common.h"
#include "classifier/svm/qpssvmlib.h"

#ifndef OCAS_MATLAB
#undef OCAS_ERRORMSG
#define OCAS_ERRORMSG(x...) { printf(x); ocas.exitflag=-2; return(ocas); }
#endif

static const uint32_t QPSolverMaxIter = 10000000;

static double *H;
static uint32_t BufSize;

/*----------------------------------------------------------------------
 Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const void *get_col( uint32_t i )
{
  return( &H[ ((uint64_t) BufSize)*i ] );
} 

/*----------------------------------------------------------------------
  Returns time of the day in seconds. 
  ----------------------------------------------------------------------*/
static double get_time()
{
	struct timeval tv;
	if (gettimeofday(&tv, NULL)==0)
		return tv.tv_sec+((double)(tv.tv_usec))/1e6;
	else
		return 0.0;
}

/*----------------------------------------------------------------------
  SVM-Ocas solver.

  Inputs:
  C  ... regularization constant.
  nData ... number of training examples
  TolRel ... stop optimization if Q_P - Q_D <= abs(Q_P)*TolRel
  TolAbs ... stop optimization if Q_P - Q_D <= TolAbs 
  QPBound ... stop optimization if Q_P <= QPBound
  _BufSize ... maximal number of constraints; currently, it also 
               determines max. num. of iterations.
  Method ... 0 ~ SVMperf/BMRM selection strategy
             1 ~ Ocas selection strategy
  
  void compute_W( double *sq_norm_W, double *dp_WoldW, double *alpha, uint32_t nSel )
  double update_W( double t )
  void add_new_cut( double *new_col_H, uint32_t *new_cut, uint32_t cut_length, uint32_t nSel )
  void compute_output( double *output )
  void qsort_index(double* value, uint32_t* index, uint32_t size)

  ----------------------------------------------------------------------*/
ocas_return_value_T svm_ocas_solver(
            double C,
            uint32_t nData, 
            double TolRel,
            double TolAbs,
            double QPBound,
            uint32_t _BufSize,
            uint8_t Method,
            void (*compute_W)(double*, double*, double*, uint32_t),
            double (*update_W)(double),
            void (*add_new_cut)(double*, uint32_t*, uint32_t, uint32_t),
            void (*compute_output)( double* ),
            void (*sort)(double*, uint32_t*, uint32_t)) 
{
  ocas_return_value_T ocas;
  double *b, *alpha, *diag_H;
  double *output, *old_output;
  double xi, sq_norm_W, QPSolverTolRel, dot_prod_WoldW, dummy, sq_norm_oldW;
  double A0, B0, Bsum, GradVal, t, t1, t2, *Ci, *Bi, *hpf;
  uint32_t *hpi;
  uint32_t nSel, cut_length;
  uint32_t i, *new_cut;
  uint16_t *I;
  int8_t qp_exitflag;

  double start_time;

  ocas.solver_time = 0;
  ocas.output_time = 0;
  ocas.sort_time = 0;

  BufSize = _BufSize;

  QPSolverTolRel = TolRel*0.1;

  /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
  H = (double*)OCAS_CALLOC(BufSize*BufSize,sizeof(double));
  if(H == NULL) OCAS_ERRORMSG("Not enough memory for matrix H.");
  
  /* bias of cutting planes */
  b = (double*)OCAS_CALLOC(BufSize,sizeof(double));
  if(b == NULL) OCAS_ERRORMSG("Not enough memory for vector b.");

  alpha = (double*)OCAS_CALLOC(BufSize,sizeof(double));
  if(alpha == NULL) OCAS_ERRORMSG("Not enough memory for vector alpha.");

  /* indices of examples which define a new cut */
  new_cut = (uint32_t*)OCAS_CALLOC(nData,sizeof(uint32_t));
  if(new_cut == NULL) OCAS_ERRORMSG("Not enough memory for vector new_cut.");

  I = (uint16_t*)OCAS_CALLOC(BufSize,sizeof(uint16_t));
  if(I == NULL) OCAS_ERRORMSG("Not enough memory for vector I.");  
  for(i=0; i< BufSize; i++) I[i] = 1;

  diag_H = (double*)OCAS_CALLOC(BufSize,sizeof(double));
  if(diag_H == NULL) OCAS_ERRORMSG("Not enough memory for vector diag_H.");

  output = (double*)OCAS_CALLOC(nData,sizeof(double));
  if(output == NULL) OCAS_ERRORMSG("Not enough memory for vector output.");

  old_output = (double*)OCAS_CALLOC(nData,sizeof(double));
  if(old_output == NULL) OCAS_ERRORMSG("Not enough memory for vector old_output.");

  /* array of hinge points used in line-serach  */
  hpf = (double*) OCAS_CALLOC(nData, sizeof(hpf[0]));
  if(hpf == NULL) OCAS_ERRORMSG("Not enough memory for array hpf.");

  hpi = (uint32_t*) OCAS_CALLOC(nData, sizeof(hpi[0]));
  if(hpi == NULL) OCAS_ERRORMSG("Not enough memory for array hpi.");

  /* vectors Ci, Bi are used in the line search procedure */
  Ci = (double*)OCAS_CALLOC(nData,sizeof(double));
  if(Ci == NULL) OCAS_ERRORMSG("Not enough memory for vector Ci.");

  Bi = (double*)OCAS_CALLOC(nData,sizeof(double));
  if(Bi == NULL) OCAS_ERRORMSG("Not enough memory for vector Bi.");
  
  nSel = 0;
  ocas.exitflag = 0;
  ocas.nIter = 0;

  /* Compute initial value of Q_P assuming that W is zero vector.*/
  sq_norm_W = 0;
  xi = nData;
  ocas.Q_P = 0.5*sq_norm_W + C*xi;
  ocas.Q_D = 0;

  /* Computed the initial cutting plane, */
  cut_length = nData;
  for(i=0; i < nData; i++)
    new_cut[i] = i;
    
  OCAS_PRINT("%4d: nsel=%d, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f, xi=%f\n",
          ocas.nIter,nSel,ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/ABS(ocas.Q_P),xi);
  
  /* main loop */
  while( ocas.exitflag == 0 )
  {
    ocas.nIter++;

    /* append new cut to buffer and update H */
    b[nSel] = -(double)cut_length;

    add_new_cut( &H[INDEX2(0,nSel,BufSize)], new_cut, cut_length, nSel );

    /* copy new added row:  H(nSel,1:nSel-1) = H(1:nSel-1:nSel)' */
    diag_H[nSel] = H[INDEX2(nSel,nSel,BufSize)];
    for(i=0; i < nSel; i++) {
      H[INDEX2(nSel,i,BufSize)] = H[INDEX2(i,nSel,BufSize)];
    }

    nSel++;    
    
    /* call inner QP solver */
    start_time = get_time();

    qp_exitflag = qpssvm_solver( &get_col, diag_H, b, C, I, alpha, 
                nSel, QPSolverMaxIter, 0.0, QPSolverTolRel, &ocas.Q_D, &dummy, 0 ); 

    ocas.solver_time += get_time() - start_time;

    ocas.Q_D = -ocas.Q_D;

    sq_norm_oldW = sq_norm_W;
    compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, nSel );
    
    /* select a new cut */
    switch( Method )
    {
      /* SVMperf ~~ BMRM strategy */
      case 0: 

        compute_output( output );

        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++)
        { 
          if(output[i] <= 0) ocas.trn_err++;
          
          if(1 - output[i] >= 0) {
            xi += 1 - output[i];
            new_cut[cut_length] = i; 
            cut_length++;
          }
        }
        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        OCAS_PRINT("%4d: Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f, xi=%f, err=%.2f%%, qp_flag=%d\n",
                  ocas.nIter,ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/ABS(ocas.Q_P), 
                  xi, 100*(double)ocas.trn_err/(double)nData, qp_exitflag);


        break;


      /* Ocas */
      case 1:

        /* Linesearch */
        A0 = sq_norm_W -2*dot_prod_WoldW + sq_norm_oldW;
        B0 = dot_prod_WoldW - sq_norm_oldW;

        memcpy( old_output, output, sizeof(double)*nData );

        start_time = get_time();
        compute_output( output );
        ocas.output_time = get_time()-start_time;

        uint32_t num_hp = 0;
        Bsum = B0;
        for(i=0; i< nData; i++) {

          Ci[i] = C*(1-old_output[i]);
          Bi[i] = C*(old_output[i] - output[i]);

          double val;
          if(Bi[i] != 0)
            val = -Ci[i]/Bi[i];
          else
            val = OCAS_PLUS_INF;
          
          if (val>0)
          {
            hpi[num_hp] = i;
            hpf[num_hp] = val;
            num_hp++;
          }
          else
            Bsum+= ABS(Bi[i]);
           
          if(Bi[i] < 0)
            Bsum += Bi[i];
        }

        start_time = get_time();
        sort(hpf, hpi, num_hp);
        ocas.sort_time += get_time() - start_time;

        t = hpf[0] - 1;
        i = 0;
        GradVal = t*A0 + Bsum;
        while( GradVal < 0 && i < num_hp && hpf[i] < OCAS_PLUS_INF ) {
          t = hpf[i];
          Bsum = Bsum + ABS(Bi[hpi[i]]);
          GradVal = t*A0 + Bsum;
          i++;
        }

        /*         t1 = MIN(MAX(t, 0.001),1.0); */             /* new W with "sanity check" */
        t1 = t;                       /* new W */
        t2 = MIN(t+(1.0-t)/10.0,1.0); /* new cutting plane */

        /* update W to be the best so far solution */
        sq_norm_W = update_W( t1 );

        /* select a new cut */
        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++ ) {

          if( 1 - (old_output[i]*(1-t2) + t2*output[i]) >=0 ) 
          {
            new_cut[cut_length] = i; 
            cut_length++;
          }

          output[i] = old_output[i]*(1-t1) + t1*output[i];

          if( 1-output[i]>=0 ) xi += 1-output[i];
          if( output[i] <= 0) ocas.trn_err++;

        }

        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        OCAS_PRINT("%4d: Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f, xi=%f, err=%.2f%%, qp_flag=%d\n",
                  ocas.nIter,ocas.Q_P,ocas.Q_D,ocas.Q_P-ocas.Q_D,(ocas.Q_P-ocas.Q_D)/ABS(ocas.Q_P),xi, 
                  100*(double)ocas.trn_err/(double)nData, qp_exitflag );

        break;
    }

    /* Stopping conditions */
    if( ocas.Q_P - ocas.Q_D <= TolRel*ABS(ocas.Q_P)) ocas.exitflag = 1; 
    if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2; 
    if( ocas.Q_P <= QPBound) ocas.exitflag = 3; 
    if(nSel >= BufSize) ocas.exitflag = -1;
         
  } /* end of the main loop */


  OCAS_FREE(H);
  OCAS_FREE(b);
  OCAS_FREE(alpha);
  OCAS_FREE(new_cut);
  OCAS_FREE(I);
  OCAS_FREE(diag_H);
  OCAS_FREE(output);
  OCAS_FREE(old_output);
  OCAS_FREE(hpf);
  OCAS_FREE(hpi);
  OCAS_FREE(Ci);
  OCAS_FREE(Bi);

  return(ocas);
}


/*----------------------------------------------------------------------
 Sort arrays value and index in asceding order according to value.
  ----------------------------------------------------------------------*/
static void swapf(double* a, double* b)
{
	double dummy=*b;
	*b=*a;
	*a=dummy;
}

static void swapi(uint32_t* a, uint32_t* b)
{
	int dummy=*b;
	*b=*a;
	*a=dummy;
}

void qsort_index(double* value, uint32_t* index, uint32_t size)
{
	if (size==2)
	{
		if (value[0] > value[1])
		{
			swapf(&value[0], &value[1]);
			swapi(&index[0], &index[1]);
		}
		return;
	}
	double split=value[size/2];

	uint32_t left=0;
	uint32_t right=size-1;

	while (left<=right)
	{
		while (value[left] < split)
			left++;
		while (value[right] > split)
			right--;

		if (left<=right)
		{
			swapf(&value[left], &value[right]);
			swapi(&index[left], &index[right]);
			left++;
			right--;
		}
	}

	if (right+1> 1)
		qsort_index(value,index,right+1);

	if (size-left> 1)
		qsort_index(&value[left],&index[left], size-left);

    return;
}
