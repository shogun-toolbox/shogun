// Math.cpp: implementation of the CMath class.
//
//////////////////////////////////////////////////////////////////////


#include "lib/Mathmatics.h"
#include "lib/io.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CMath math;
#ifndef NO_LOG_CACHE
//gene/math specific constants
#ifdef DEBUG
 #define MAX_LOG_TABLE_SIZE 10*1024*1024
 #define LOG_TABLE_PRECISION 1e-6
#else
 #define MAX_LOG_TABLE_SIZE 123*1024*1024
 #define LOG_TABLE_PRECISION 1e-15
#endif

int CMath::LOGRANGE            = 0; // range for logtable: log(1+exp(x))  -25 <= x <= 0
int CMath::LOGACCURACY         = 0; // 100000 steps per integer
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
#ifndef NO_LOG_CACHE
    LOGRANGE=CMath::determine_logrange();
    LOGACCURACY=CMath::determine_logaccuracy(LOGRANGE);
    CIO::message("Initializing log-table (size=%i*%i*%i=%2.1fMB) ...",LOGRANGE,LOGACCURACY,sizeof(REAL),LOGRANGE*LOGACCURACY*sizeof(REAL)/(1024.0*1024.0)) ;
   
    CMath::logtable=new REAL[LOGRANGE*LOGACCURACY];
    init_log_table();
    CIO::message("Done.\n") ;
#endif 
}

CMath::~CMath()
{
#ifndef NO_LOG_CACHE
	delete[] logtable;
#endif
}

#ifndef NO_LOG_CACHE
int CMath::determine_logrange()
{
    int i;
    REAL acc=0;
    for (i=0; i<50; i++)
    {
	acc=((REAL)log(1+((REAL)exp(-REAL(i)))));
	if (acc<=(REAL)LOG_TABLE_PRECISION)
	    break;
    }

    CIO::message("determined range for x in table log(1+exp(-x)) is:%d (error:%G)\n",i,acc);
    return i;
}

int CMath::determine_logaccuracy(int range)
{
    range=MAX_LOG_TABLE_SIZE/range/((int)sizeof(REAL));
    CIO::message("determined accuracy for x in table log(1+exp(-x)) is:%d (error:%G)\n",range,1.0/(double) range);
    return range;
}

//init log table of form log(1+exp(x))
void CMath::init_log_table()
{
  for (int i=0; i< LOGACCURACY*LOGRANGE; i++)
    logtable[i]=log(1+exp(REAL(-i)/REAL(LOGACCURACY)));
}
#endif

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

void CMath::qsort(REAL* output, int size)
{
    REAL split=output[(RAND_MAX+1+size*rand())/(RAND_MAX+1)];
    //REAL split=output[size/2];

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
int CMath::calcroc(REAL* fp, REAL* tp, REAL* output, int* label, int size, int& possize, int& negsize, FILE* rocfile)
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
    REAL* negout=output;
    REAL* posout=&output[left];
    //int* neglab=label;
    //int* poslab=&label[left];

    qsort(negout, negsize);
    qsort(posout, possize);
   
    REAL minimum=min(negout[0], posout[0]);
    //REAL maximum=max(negout[negsize-1], posout[possize-1]);

    for (int i=0; i<size; i++)
    {
	fp[i]=1.0;
	tp[i]=1.0;
    }

    int posidx=0;
    int negidx=0;
    int iteration=1;

    REAL treshhold=minimum;
    int returnidx=-1;

    REAL minerr=10;

    while (iteration < size)
    {
	tp[iteration]=(possize-posidx)/(REAL (possize));
	fp[iteration]=(negsize-negidx)/(REAL (negsize));
   
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
	fwrite(fp, sizeof(REAL), size, rocfile);
	fwrite(tp, sizeof(REAL), size, rocfile);
    }
    return returnidx;
}

unsigned int CMath::crc32(unsigned char *data, int len)
{
    unsigned int        result;
    int                 i,j;
    unsigned char       octet;

    result = 0-1;

    for (i=0; i<len; i++)
    {
	octet = *(data++);
	for (j=0; j<8; j++)
	{
	    if ((octet >> 7) ^ (result >> 31))
	    {
		result = (result << 1) ^ 0x04c11db7;
	    }
	    else
	    {
		result = (result << 1);
	    }
	    octet <<= 1;
	}
    }

    return ~result; 
}
