/*
 * Compute the local alignment kernel
 *
 * Largely based on LAkernel.c (version 0.3)
 *
 * Copyright 2003 Jean-Philippe Vert
 * Copyright 2005 Jean-Philippe Vert, Hiroto Saigo
 *
 * Shogun specific adjustments Written (W) 2007-2008,2010 Soeren Sonnenburg
 *
 * Reference:
 * H. Saigo, J.-P. Vert, T. Akutsu and N. Ueda, "Protein homology
 * detection using string alignment kernels", Bioinformatics,
 * vol.20, p.1682-1689, 2004.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 */

#include <stdlib.h>
#include <stdio.h>
#include <shogun/mathematics/Math.h>
#include <ctype.h>
#include <string.h>
#include <shogun/kernel/string/LocalAlignmentStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>

using namespace shogun;

/****************/
/* The alphabet */
/****************/

const int32_t NAA=20;                                  /* Number of amino-acids */
const int32_t NLET=26;                                 /* Number of letters in the alphabet */
const char* CLocalAlignmentStringKernel::aaList="ARNDCQEGHILKMFPSTWYV";    /* The list of amino acids */

/*****************/
/* SW parameters */
/*****************/

/* mutation matrix */
const int32_t CLocalAlignmentStringKernel::blosum[] = {
  6,
 -2,   8,
 -2,  -1,   9,
  -3,  -2,   2,   9,
  -1,  -5,  -4,  -5,  13,
  -1,   1,   0,   0,  -4,   8,
  -1,   0,   0,   2,  -5,   3,   7,
  0,  -3,  -1,  -2,  -4,  -3,  -3,   8,
  -2,   0,   1,  -2,  -4,   1,   0,  -3,  11,
  -2,  -5,  -5,  -5,  -2,  -4,  -5,  -6,  -5,   6,
  -2,  -3,  -5,  -5,  -2,  -3,  -4,  -5,  -4,   2,   6,
  -1,   3,   0,  -1,  -5,   2,   1,  -2,  -1,  -4,  -4,   7,
  -1,  -2,  -3,  -5,  -2,  -1,  -3,  -4,  -2,   2,   3,  -2,   8,
  -3,  -4,  -5,  -5,  -4,  -5,  -5,  -5,  -2,   0,   1,  -5,   0,   9,
  -1,  -3,  -3,  -2,  -4,  -2,  -2,  -3,  -3,  -4,  -4,  -2,  -4,  -5,  11,
  2,  -1,   1,   0,  -1,   0,   0,   0,  -1,  -4,  -4,   0,  -2,  -4,  -1,   6,
  0,  -2,   0,  -2,  -1,  -1,  -1,  -2,  -3,  -1,  -2,  -1,  -1,  -3,  -2,   2,   7,
  -4,  -4,  -6,  -6,  -3,  -3,  -4,  -4,  -4,  -4,  -2,  -4,  -2,   1,  -6,  -4,  -4,  16,
  -3,  -3,  -3,  -5,  -4,  -2,  -3,  -5,   3,  -2,  -2,  -3,  -1,   4,  -4,  -3,  -2,   3,  10,
  0,  -4,  -4,  -5,  -1,  -3,  -4,  -5,  -5,   4,   1,  -3,   1,  -1,  -4,  -2,   0,  -4,  -2,   6};

/* Index corresponding to the (i,j) entry (i,j=0..19) in the blosum matrix */
static int32_t BINDEX(int32_t i, int32_t j)
{
	return (((i)>(j))?(j)+(((i)*(i+1))/2):(i)+(((j)*(j+1))/2));
}

/*********************
 * Kernel parameters *
 *********************/

const float64_t SCALING=0.1;           /* Factor to scale all SW parameters */

/* If you want to compute the sum over all local alignments (to get a valid kernel), uncomment the following line : */
/* If x=log(a) and y=log(b), compute log(a+b) : */
/*
#define LOGP(x,y) (((x)>(y))?(x)+log1p(exp((y)-(x))):(y)+log1p(exp((x)-(y))))
*/

#define LOGP(x,y) LogSum(x,y)

/* OR if you want to compute the score of the best local alignment (to get the SW score by Viterbi), uncomment the following line : */
/*
#define LOGP(x,y) (((x)>(y))?(x):(y))
*/

/* Usefule constants */
const float64_t LOG0=-10000;          /* log(0) */
const float64_t INTSCALE=1000.0;      /* critical for speed and precise computation*/

int32_t CLocalAlignmentStringKernel::logsum_lookup[LOGSUM_TBL];

CLocalAlignmentStringKernel::CLocalAlignmentStringKernel(int32_t size)
: CStringKernel<char>(size)
{
	init();
	init_static_variables();
}

CLocalAlignmentStringKernel::CLocalAlignmentStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r,
	float64_t opening, float64_t extension)
: CStringKernel<char>()
{
	init();
	m_opening=opening;
	m_extension=extension;
	init_static_variables();
	init(l, r);
}

CLocalAlignmentStringKernel::~CLocalAlignmentStringKernel()
{
	cleanup();
}

bool CLocalAlignmentStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	initialized=true;
	return init_normalizer();
}

void CLocalAlignmentStringKernel::cleanup()
{
	SG_FREE(scaled_blosum);
	scaled_blosum=NULL;

	SG_FREE(isAA);
	isAA=NULL;
	SG_FREE(aaIndex);
	aaIndex=NULL;

	CKernel::cleanup();
}

/* LogSum - default log funciotion. fast, but not exact */
/* LogSum2 - precise, but slow. Note that these two functions need different figure types  */

void CLocalAlignmentStringKernel::init_logsum(){
	int32_t i;
	for (i=0; i<LOGSUM_TBL; i++)
		logsum_lookup[i]=int32_t(INTSCALE*
				(log(1.+exp((float32_t)-i/INTSCALE))));
}

int32_t CLocalAlignmentStringKernel::LogSum(int32_t p1, int32_t p2)
{
	int32_t diff;
	static int32_t firsttime=1;

	if (firsttime)
	{
		init_logsum();
		firsttime =0;
	}

	diff=p1-p2;
	if (diff>=LOGSUM_TBL)
		return p1;
	else if (diff<=-LOGSUM_TBL)
		return p2;
	else if (diff>0)
		return p1+logsum_lookup[diff];
	else
		return p2+logsum_lookup[-diff];
}


float32_t CLocalAlignmentStringKernel::LogSum2(float32_t p1, float32_t p2)
{
	if (p1>p2)
		return (p1-p2>50.) ? p1 : p1+log(1.+exp(p2-p1));
	else
		return (p2-p1>50.) ? p2 : p2+log(1.+exp(p1-p2));
}


void CLocalAlignmentStringKernel::init_static_variables()
     /* Initialize all static variables. This function should be called once before computing the first pair HMM score */
{
	register int32_t i;

	/* Initialization of the array which gives the position of each amino-acid in the set of amino-acid */
	aaIndex = SG_CALLOC(int32_t, NLET);
	for (i=0;i<NAA;i++)
		aaIndex[aaList[i]-'A']=i;

	/* Initialization of the array which indicates whether a char is an amino-acid */
	isAA = SG_CALLOC(int32_t, 256);
	for (i=0;i<NAA;i++)
		isAA[(int32_t)aaList[i]]=1;

	/* Scale the blossum matrix */
	for (i=0 ; i<NAA*(NAA+1)/2; i++)
		scaled_blosum[i]=(int32_t)floor(blosum[i]*SCALING*INTSCALE);


	/* Scale of gap penalties */
	m_opening=(int32_t)floor(m_opening*SCALING*INTSCALE);
	m_extension=(int32_t)floor(m_extension*SCALING*INTSCALE);
}



/* Implementation of the
 * convolution kernel which generalizes the Smith-Waterman algorithm
 */
float64_t CLocalAlignmentStringKernel::LAkernelcompute(
	int32_t* aaX, int32_t* aaY, /* the two amino-acid sequences (as sequences of indexes in [0..NAA-1] indicating the position of the amino-acid in the variable 'aaList') */
	int32_t nX, int32_t nY /* the lengths of both sequences */)
{
	register int32_t
	i,j,                /* loop indexes */
	cur, old,           /* to indicate the array to use (0 or 1) */
	curpos, frompos;    /* position in an array */

	int32_t
	*logX,           /* arrays to store the log-values of each state */
	*logY,
	*logM,
	*logX2,
	*logY2;

	int32_t aux, aux2;/* , aux3 , aux4 , aux5;*/
	int32_t cl;/* length of a column for the dynamic programming */

	/*
	printf("now computing pairHMM between %d and %d:\n",nX,nY);
	for (i=0;i<nX;printf("%d ",aaX[i++]));
	printf("\n and \n");
	for (i=0;i<nY;printf("%d ",aaY[i++]));
	printf("\n");
	*/

	/* Initialization of the arrays */
	/* Each array stores two successive columns of the (nX+1)x(nY+1) table used in dynamic programming */
	cl=nY+1;           /* each column stores the positions in the aaY sequence, plus a position at zero */

	logM=SG_CALLOC(int32_t, 2*cl);
	logX=SG_CALLOC(int32_t, 2*cl);
	logY=SG_CALLOC(int32_t, 2*cl);
	logX2=SG_CALLOC(int32_t, 2*cl);
	logY2=SG_CALLOC(int32_t, 2*cl);

	/************************************************/
	/* First iteration : initialization of column 0 */
	/************************************************/
	/* The log=proabilities of each state are initialized for the first column (x=0,y=0..nY) */

	for (j=0; j<cl; j++) {
		logM[j]=LOG0;
		logX[j]=LOG0;
		logY[j]=LOG0;
		logX2[j]=LOG0;
		logY2[j]=LOG0;
	}

	/* Update column order */
	cur=1;      /* Indexes [0..cl-1] are used to process the next column */
	old=0;      /* Indexes [cl..2*cl-1] were used for column 0 */


	/************************************************/
	/* Next iterations : processing columns 1 .. nX */
	/************************************************/

	/* Main loop to vary the position in aaX : i=1..nX */
	for (i=1; i<=nX; i++) {

		/* Special update for positions (i=1..nX,j=0) */
		curpos=cur*cl;                  /* index of the state (i,0) */
		logM[curpos]=LOG0;
		logX[curpos]=LOG0;
		logY[curpos]=LOG0;
		logX2[curpos]=LOG0;
		logY2[curpos]=LOG0;

		/* Secondary loop to vary the position in aaY : j=1..nY */
		for (j=1; j<=nY; j++) {

			curpos=cur*cl+j;            /* index of the state (i,j) */

			/* Update for states which emit X only */
			/***************************************/

			frompos=old*cl+j;            /* index of the state (i-1,j) */

			/* State RX */
			logX[curpos]=LOGP(-m_opening+logM[frompos], -m_extension+logX[frompos]);
			/*      printf("%.5f\n",logX[curpos]);*/
			/*      printf("%.5f\n",logX_B[curpos]);*/
			/* State RX2 */
			logX2[curpos]=LOGP(logM[frompos], logX2[frompos]);

			/* Update for states which emit Y only */
			/***************************************/

			frompos=cur*cl+j-1;          /* index of the state (i,j-1) */

			/* State RY */
			aux=LOGP(-m_opening+logM[frompos], -m_extension+logY[frompos]);
			logY[curpos] = LOGP(aux, -m_opening+logX[frompos]);

			/* State RY2 */
			aux=LOGP(logM[frompos], logY2[frompos]);
			logY2[curpos]=LOGP(aux, logX2[frompos]);

			/* Update for states which emit X and Y */
			/****************************************/

			frompos=old*cl+j-1;          /* index of the state (i-1,j-1) */

			aux=LOGP(logX[frompos], logY[frompos]);
			aux2=LOGP(0, logM[frompos]);
			logM[curpos]=LOGP(aux, aux2)+scaled_blosum[BINDEX(aaX[i-1], aaY[j-1])];

			/*
			printf("i=%d , j=%d\nM=%.5f\nX=%.5f\nY=%.5f\nX2=%.5f\nY2=%.5f\n",i,j,logM[curpos],logX[curpos],logY[curpos],logX2[curpos],logY2[curpos]);
			*/

		}  /* end of j=1:nY loop */


		/* Update the culumn order */
		cur=1-cur;
		old=1-old;

	}  /* end of j=1:nX loop */


	/* Termination */
	/***************/

	curpos=old*cl+nY;                /* index of the state (nX,nY) */
	aux=LOGP(logX2[curpos], logY2[curpos]);
	aux2=LOGP(0, logM[curpos]);
	/*  kernel_value = LOGP( aux , aux2 );*/

	/* Memory release */
	SG_FREE(logM);
	SG_FREE(logX);
	SG_FREE(logY);
	SG_FREE(logX2);
	SG_FREE(logY2);

	/* Return the logarithm of the kernel */
	return (float32_t)LOGP(aux,aux2)/INTSCALE;
}

/********************/
/* Public functions */
/********************/


/* Return the log-probability of two sequences x and y under a pair HMM model */
/* x and y are strings of aminoacid letters, e.g., "AABRS" */
float64_t CLocalAlignmentStringKernel::compute(int32_t idx_x, int32_t idx_y)
{
	int32_t *aax, *aay;  /* to convert x and y into sequences of amino-acid indexes */
	int32_t lx=0, ly=0;       /* lengths of x and y */
	int32_t i, j;

	bool free_x, free_y;
	char* x=((CStringFeatures<char>*) lhs)->get_feature_vector(idx_x, lx, free_x);
	char* y=((CStringFeatures<char>*) rhs)->get_feature_vector(idx_y, ly, free_y);
	ASSERT(x && y)

	if ( (lx<1) || (ly<1) )
		SG_ERROR("empty chain")

	/* Create aax and aay */

	aax = SG_CALLOC(int32_t, lx);
	aay = SG_CALLOC(int32_t, ly);

	/* Extract the characters corresponding to aminoacids and keep their indexes */

	j=0;
	for (i=0; i<lx; i++)
		if (isAA[toupper(x[i])])
			aax[j++]=aaIndex[toupper(x[i])-'A'];
	lx=j;
	j=0;
	for (i=0; i<ly; i++)
		if (isAA[toupper(y[i])])
			aay[j++]=aaIndex[toupper(y[i])-'A'];
	ly=j;


	/* Compute the pair HMM score */
	float64_t result=LAkernelcompute(aax, aay, lx, ly);

	/* Release memory */
	SG_FREE(aax);
	SG_FREE(aay);

	((CStringFeatures<char>*)lhs)->free_feature_vector(x, idx_x, free_x);
	((CStringFeatures<char>*)rhs)->free_feature_vector(y, idx_y, free_y);

	return result;
}

void CLocalAlignmentStringKernel::init()
{
	set_normalizer(new CSqrtDiagKernelNormalizer());

	initialized=false;
	isAA=NULL;
	aaIndex=NULL;

	m_opening=10;
	m_extension=2;

	scaled_blosum=SG_CALLOC(int32_t, sizeof(blosum));
	init_logsum();

	SG_ADD(&initialized, "initialized", "If kernel is initalized.",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_opening, "opening", "Opening gap opening penalty.", MS_AVAILABLE);
	SG_ADD(&m_extension, "extension", "Extension gap extension penalty.",
		MS_AVAILABLE);
}
