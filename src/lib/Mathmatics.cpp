// Math.cpp: implementation of the CMath class.
//
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include "Mathmatics.h"
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CMath math;

//gene/math specific constants
const int CMath::LOGRANGE            = 30 ;		// range for logtable: log(1+exp(x))  -25 <= x <= 0
#ifndef _DEBUG
const int CMath::LOGACCURACY         = 100000;	        // 100000 steps per integer
#else
const int CMath::LOGACCURACY         = 40000;	        // 100000 steps per integer
#endif

#ifdef PATHDEBUG
const REAL CMath::INFTY            =  1e11;	        // infinity
#else
const REAL CMath::INFTY            =  -log(0.0);	// infinity
#endif
const REAL CMath::ALMOST_NEG_INFTY =  -1000;	

CMath::CMath()
{
    srand(time(NULL));
    fprintf(stdout,"Initializing log-table (size=%i*%i*%i=%2.1fMB) ...",LOGRANGE,LOGACCURACY,sizeof(REAL),LOGRANGE*LOGACCURACY*sizeof(REAL)/(1024.0*1024.0)) ;
    fflush(stdout);
    CMath::logtable=new REAL[LOGRANGE*LOGACCURACY];
    init_log_table();
    fprintf(stdout,"Done.\n") ;
    fflush(stdout);
}

CMath::~CMath()
{
	delete[] logtable;
}

//init log table of form log(1+exp(x))
void CMath::init_log_table()
{
  for (int i=0; i< LOGACCURACY*LOGRANGE; i++)
    logtable[i]=log(1+exp(REAL(-i)/REAL(LOGACCURACY)));
}

void CMath::sort(int *a, int cols, int sort_col)
{
  int changed=1;
  if (a[0]==-1) return ;
  while (changed)
    {
      changed=0; int i=0 ;
      while ((a[(i+1)*cols]!=-1) && (a[(i+1)*cols+1]!=-1)) // to be sure
	{
	  if (a[i*cols+sort_col]>a[(i+1)*cols+sort_col])
	    {
	      for (int j=0; j<cols; j++)
		swap(a[i*cols+j],a[(i+1)*cols+j]) ;
	      changed=1 ;
	    } ;
	  i++ ;
	} ;
    } ;
} 

void CMath::qsort(double* output, int size)
{
    double split=output[(RAND_MAX+1+size*rand())/(RAND_MAX+1)];
    //double split=output[size/2];

    int left=0;
    int right=size-1;
    
    while (left<=right)
    {
	while (output[left] < split)
	    left++;
	while (output[right] > split)
	    right--;

	if (left<=right)
	{
	    swap(output[left],output[right]);
	    left++;
	    right--;
	}
    }

    if (right+1> 1)
	qsort(output,right+1);
    
    if (size-left> 1)
	qsort(&output[left],size-left);
}

//plot x- axis false positives (fp) 1-Specificity
//plot y- axis true positives (tp) Sensitivity
int CMath::calcroc(double* fp, double* tp, double* output, int* label, int size, int& possize, int& negsize, FILE* rocfile)
{
    int left=0;
    int right=size-1;

    //-1 labels first
    while (left<right)
    {
	while (label[left] < 0 )
	    left++;
	while ((label[right] > 0) && (left<right))	//warning: label must be either -1 or +1
	    right--;
	
	swap(output[left],output[right]);
	swap(label[left], label[right]);
    }
   
    negsize=left;
    possize=size-left;
    double* negout=output;
    double* posout=&output[left];
    int* neglab=label;
    int* poslab=&label[left];

    qsort(negout, negsize);
    qsort(posout, possize);
   
    double minimum=min(negout[0], posout[0]);
    double maximum=max(negout[negsize-1], posout[possize-1]);

    for (int i=0; i<size; i++)
    {
	fp[i]=1.0;
	tp[i]=1.0;
    }

    int posidx=0;
    int negidx=0;
    int iteration=1;

    double treshhold=minimum;
    int returnidx=-1;

    double minerr=10;

    while (iteration < size)
    {
	tp[iteration]=(possize-posidx)/(double (possize));
	fp[iteration]=(negsize-negidx)/(double (negsize));
   
	if (minerr > negsize*fp[iteration]/size+(1-tp[iteration])*possize/size )
	{
	    minerr=negsize*fp[iteration]/size+(1-tp[iteration])*possize/size;
	    returnidx=iteration;
	}

	if ( (posidx>=possize) || (negidx>=negsize) )
	{
	    if (posidx < possize)
	    {
		treshhold=posout[posidx];
		posidx++;
	    }
	    else if (negidx < negsize)
	    {
		treshhold=negout[negidx];
		negidx++;
	    }
	}
	else 
	{
	    if ( posout[posidx] < negout[negidx] )
	    {
		treshhold=posout[posidx];
		posidx++;
	    }
	    else if (posout[posidx] == negout[negidx])
	    {
		treshhold=negout[negidx];
		negidx++;
		posidx++;
	    }
	    else
	    {
		treshhold=negout[negidx];
		negidx++;
	    }
	}
	iteration++;
    }

    if (rocfile)
    {
	const char id[]="ROC";
	fwrite(id, sizeof(char), sizeof(id), rocfile);
	fwrite(fp, sizeof(double), size, rocfile);
	fwrite(tp, sizeof(double), size, rocfile);
    }
    return returnidx;
}
