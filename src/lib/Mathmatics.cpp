// Math.cpp: implementation of the CMath class.
//
//////////////////////////////////////////////////////////////////////


#include "lib/Mathmatics.h"
#include "lib/io.h"

#include <sys/time.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CMath math;
#ifdef USE_LOGCACHE
//gene/math specific constants
#ifdef USE_HMMDEBUG
#define MAX_LOG_TABLE_SIZE 10*1024*1024
#define LOG_TABLE_PRECISION 1e-6
#else
#define MAX_LOG_TABLE_SIZE 123*1024*1024
#define LOG_TABLE_PRECISION 1e-15
#endif

INT CMath::LOGACCURACY         = 0; // 100000 steps per integer
#endif

INT CMath::LOGRANGE            = 0; // range for logtable: log(1+exp(x))  -25 <= x <= 0

#ifdef USE_PATHDEBUG
const REAL CMath::INFTY            =  1e11;	        // infinity
#else
const REAL CMath::INFTY            =  -log(0.0);	// infinity
#endif
const REAL CMath::ALMOST_NEG_INFTY =  -1000;	

CHAR CMath::rand_state[256];

CMath::CMath()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	UINT seed=(UINT) (4223517*getpid()*tv.tv_sec*tv.tv_usec);
	initstate(seed, CMath::rand_state, sizeof(CMath::rand_state));
	CIO::message(M_INFO, "seeding random number generator with %u\n", seed);

#ifdef USE_LOGCACHE
    LOGRANGE=CMath::determine_logrange();
    LOGACCURACY=CMath::determine_logaccuracy(LOGRANGE);
    CIO::message(M_INFO, "Initializing log-table (size=%i*%i*%i=%2.1fMB) ...",LOGRANGE,LOGACCURACY,sizeof(REAL),LOGRANGE*LOGACCURACY*sizeof(REAL)/(1024.0*1024.0)) ;
   
    CMath::logtable=new REAL[LOGRANGE*LOGACCURACY];
    init_log_table();
    CIO::message(M_INFO, "Done.\n") ;
#else
	INT i=0;
	while ((REAL)log(1+((REAL)exp(-REAL(i)))))
		i++;
    CIO::message(M_INFO, "determined range for x in log(1+exp(-x)) is:%d\n", i);
	LOGRANGE=i;
#endif 
}

CMath::~CMath()
{
#ifdef USE_LOGCACHE
	delete[] logtable;
#endif
}

#ifdef USE_LOGCACHE
INT CMath::determine_logrange()
{
    INT i;
    REAL acc=0;
    for (i=0; i<50; i++)
    {
	acc=((REAL)log(1+((REAL)exp(-REAL(i)))));
	if (acc<=(REAL)LOG_TABLE_PRECISION)
	    break;
    }

    CIO::message(M_INFO, "determined range for x in table log(1+exp(-x)) is:%d (error:%G)\n",i,acc);
    return i;
}

INT CMath::determine_logaccuracy(INT range)
{
    range=MAX_LOG_TABLE_SIZE/range/((int)sizeof(REAL));
    CIO::message(M_INFO, "determined accuracy for x in table log(1+exp(-x)) is:%d (error:%G)\n",range,1.0/(double) range);
    return range;
}

//init log table of form log(1+exp(x))
void CMath::init_log_table()
{
  for (INT i=0; i< LOGACCURACY*LOGRANGE; i++)
    logtable[i]=log(1+exp(REAL(-i)/REAL(LOGACCURACY)));
}
#endif

void CMath::sort(INT *a, INT cols, INT sort_col)
{
  INT changed=1;
  if (a[0]==-1) return ;
  while (changed)
  {
      changed=0; INT i=0 ;
      while ((a[(i+1)*cols]!=-1) && (a[(i+1)*cols+1]!=-1)) // to be sure
	  {
		  if (a[i*cols+sort_col]>a[(i+1)*cols+sort_col])
		  {
			  for (INT j=0; j<cols; j++)
				  CMath::swap(a[i*cols+j],a[(i+1)*cols+j]) ;
			  changed=1 ;
		  } ;
		  i++ ;
	  } ;
  } ;
} 

void CMath::sort(REAL *a, INT* idx, INT N) 
{

	INT changed=1;
	while (changed)
	{
		changed=0;
		for (INT i=0; i<N-1; i++)
		{
			if (a[i]>a[i+1])
			{
				swap(a[i],a[i+1]) ;
				swap(idx[i],idx[i+1]) ;
				changed=1 ;
			} ;
		} ;
	} ;
	 
} 


void CMath::qsort(WORD* output, INT size)
{
		if (size==2)
		{
				if (output[0] > output [1])
						swap(output[0],output[1]);
		}
		else
		{
				REAL split=output[(size*rand())/(RAND_MAX+1)];
				//REAL split=output[size/2];

				INT left=0;
				INT right=size-1;

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
}

void CMath::qsort(REAL* output, INT size)
{
		if (size==2)
		{
				if (output[0] > output [1])
						CMath::swap(output[0],output[1]);
		}
		else
		{
				REAL split=output[(size*rand())/(RAND_MAX+1)];
				//REAL split=output[size/2];

				INT left=0;
				INT right=size-1;

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
}

template <class T>
void CMath::qsort(REAL* output, T* index, INT size)
{
	if (size==2)
	{
		if (output[0] > output [1]){
			swap(output[0],output[1]);
			swap(index[0],index[1]);
		}
		
	}
	else
	{
		REAL split=output[(size*rand())/(RAND_MAX+1)];
		//REAL split=output[size/2];
		
		INT left=0;
		INT right=size-1;
		
		while (left<=right)
		{
			while (output[left] < split)
				left++;
			while (output[right] > split)
				right--;
			
			if (left<=right)
			{
				swap(output[left],output[right]);
				swap(index[left],index[right]);
				left++;
				right--;
			}
		}
		
		if (right+1> 1)
			qsort(output,index,right+1);
		
		if (size-left> 1)
			qsort(&output[left],&index[left], size-left);
	}
}

void CMath::qsort_backward(REAL* output, INT* index, INT size)
{
	if (size==2)
	{
		if (output[0] < output [1]){
			swap(output[0],output[1]);
			swap(index[0],index[1]);
		}
		
	}
	else
	{
		REAL split=output[(size*rand())/(RAND_MAX+1)];
		//REAL split=output[size/2];
		
		INT left=0;
		INT right=size-1;
		
		while (left<=right)
		{
			while (output[left] > split)
				left++;
			while (output[right] < split)
				right--;
			
			if (left<=right)
			{
				swap(output[left],output[right]);
				swap(index[left],index[right]);
				left++;
				right--;
			}
		}
		
		if (right+1> 1)
			qsort(output,index,right+1);
		
		if (size-left> 1)
			qsort(&output[left],&index[left], size-left);
	}
}

//plot x- axis false positives (fp) 1-Specificity
//plot y- axis true positives (tp) Sensitivity
INT CMath::calcroc(REAL* fp, REAL* tp, REAL* output, INT* label, INT& size, INT& possize, INT& negsize, REAL& tresh, FILE* rocfile)
{
	INT left=0;
	INT right=size-1;
	INT i;

	for (i=0; i<size; i++)
	{
		if (!(label[i]== -1 || label[i]==1))
			return -1;
	}

	//sort data such that -1 labels first +1 behind
	while (left<right)
	{
		while ((label[left] < 0) && (left<right))
			left++;
		while ((label[right] > 0) && (left<right))	//warning: label must be either -1 or +1
			right--;

		swap(output[left],output[right]);
		swap(label[left], label[right]);
	}

	// neg/pos sizes
	negsize=left;
	possize=size-left;
	REAL* negout=output;
	REAL* posout=&output[left];

	// sort the pos and neg outputs separately
	qsort(negout, negsize);
	qsort(posout, possize);

	// set minimum+maximum values for decision-treshhold
	REAL minimum=min(negout[0], posout[0]);
	REAL maximum=minimum;
	if (negsize>0)
		maximum=max(maximum,negout[negsize-1]);
	if (possize>0)
		maximum=max(maximum,posout[possize-1]);

	REAL treshhold=minimum;
	REAL old_treshhold=treshhold;

	//clear array.
	for (i=0; i<size; i++)
	{
		fp[i]=1.0;
		tp[i]=1.0;
	}

	//start with fp=1.0 tp=1.0 which is posidx=0, negidx=0
	//everything right of {pos,neg}idx is considered to beLONG to +1
	INT posidx=0;
	INT negidx=0;
	INT iteration=1;
	INT returnidx=-1;

	REAL minerr=10;

	while (iteration < size && treshhold<=maximum)
	{
		old_treshhold=treshhold;

		while (treshhold==old_treshhold && treshhold<=maximum)
		{
			if (posidx<possize && negidx<negsize)
			{
				if (posout[posidx]<negout[negidx])
				{
					if (posout[posidx]==treshhold)
						posidx++;
					else
						treshhold=posout[posidx];
				}
				else
				{
					if (negout[negidx]==treshhold)
						negidx++;
					else
						treshhold=negout[negidx];
				}
			}
			else
			{
				if (posidx>=possize && negidx<negsize-1)
				{
					negidx++;
					treshhold=negout[negidx];
				}
				else if (negidx>=negsize && posidx<possize-1)
				{
					posidx++;
					treshhold=posout[posidx];
				}
				else if (negidx<negsize && treshhold!=negout[negidx])
					treshhold=negout[negidx];
				else if (posidx<possize && treshhold!=posout[posidx])
					treshhold=posout[posidx];
				else
				{
					treshhold=2*(maximum+100); // force termination
					posidx=possize;
					negidx=negsize;
					break;
				}
			}
		}

		//calc tp,fp
		tp[iteration]=(possize-posidx)/(REAL (possize));
		fp[iteration]=(negsize-negidx)/(REAL (negsize));

		//choose poINT with minimal error
		if (minerr > negsize*fp[iteration]/size+(1-tp[iteration])*possize/size )
		{
			minerr=negsize*fp[iteration]/size+(1-tp[iteration])*possize/size;
			tresh=(old_treshhold+treshhold)/2;
			returnidx=iteration;
		}

		iteration++;
	}

	// set new size
	size=iteration;

	if (rocfile)
	{
		const CHAR id[]="ROC";
		fwrite(id, sizeof(char), sizeof(id), rocfile);
		fwrite(fp, sizeof(REAL), size, rocfile);
		fwrite(tp, sizeof(REAL), size, rocfile);
	}

	return returnidx;
}

UINT CMath::crc32(BYTE *data, INT len)
{
    UINT        result;
    INT                 i,j;
    BYTE       octet;

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

double CMath::mutual_info(REAL* p1, REAL* p2, INT len)
{
	double e=0;

	for (INT i=0; i<len; i++)
		for (INT j=0; j<len; j++)
			e+=exp(p2[j*len+i])*(p2[j*len+i]-p1[i]-p1[j]);

	return e;
}

double CMath::relative_entropy(REAL* p, REAL* q, INT len)
{
	double e=0;

	for (INT i=0; i<len; i++)
		e+=exp(p[i])*(p[i]-q[i]);

	return e;
}

double CMath::entropy(REAL* p, INT len)
{
	double e=0;

	for (INT i=0; i<len; i++)
		e-=exp(p[i])*p[i];

	return e;
}
