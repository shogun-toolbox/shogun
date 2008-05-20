/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Alexander Binder
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "PyramidChi2.h"
#include "lib/common.h"
#include "kernel/GaussianKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CPyramidChi2::CPyramidChi2(INT size, DREAL width2, INT* pyramidlevels2,INT
		numlevels2, INT  numbinsinhistogram2, DREAL* weights2, INT numweights2)
: CSimpleKernel<DREAL>(size), width(width2), pyramidlevels(NULL),
	numlevels(numlevels2), weights(NULL), numweights(numweights2)
{
	pyramidlevels=new INT[numlevels];
	for(INT i=0; i<numlevels;++i )
		pyramidlevels[i]=pyramidlevels2[i];
	
	numbinsinhistogram=numbinsinhistogram2;
	
	weights=new DREAL[numweights];
	for(INT i=0; i<numweights;++i )
		weights[i]=weights2[i];
	
	if (!sanitycheck_weak())
		SG_ERROR("CPyramidChi2::CPyramidChi2(... first constructor): false==sanitycheck_weak() occurred! Someone messed up the initializing of the kernel.\0");
}

void CPyramidChi2::cleanup()
{
	//weights.clear();
	//pyramidlevels.clear();
	numlevels=-1;
	numweights=-1;
	numbinsinhistogram=-1;
	//sanitycheckbit=false;
	
	delete[] pyramidlevels;
	pyramidlevels=NULL;
	delete[] weights;
	weights=NULL;

}

bool CPyramidChi2::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleKernel<DREAL>::init(l, r);
	return result;
}

CPyramidChi2::CPyramidChi2(CRealFeatures* l, CRealFeatures* r, INT size, DREAL width2,
		INT* pyramidlevels2,INT numlevels2,
		INT  numbinsinhistogram2, DREAL* weights2,INT numweights2) :
	CSimpleKernel<DREAL>(size), width(width2),pyramidlevels(NULL),numlevels(numlevels2),weights(NULL),numweights(numweights2)
{
	pyramidlevels=new INT[numlevels];
	for(INT i=0; i<numlevels;++i )
		pyramidlevels[i]=pyramidlevels2[i];
	
	numbinsinhistogram=numbinsinhistogram2;
	
	weights=new DREAL[numweights];
	for(INT i=0; i<numweights;++i )
		weights[i]=weights2[i];
	
	if(!sanitycheck_weak())
		SG_ERROR("CPyramidChi2::CPyramidChi2(... second constructor): false==sanitycheck_weak() occurred! Someone messed up with initializing the kernel.\0");

	init(l, r);
}

CPyramidChi2::~CPyramidChi2()
{
	cleanup();
}

bool CPyramidChi2::load_init(FILE* src)
{
	return (false);
}

bool CPyramidChi2::save_init(FILE* dest)
{
	return (false);
}


bool CPyramidChi2::sanitycheck_weak()
{
	if (numbinsinhistogram<=0)
	{
		SG_ERROR("bool CPyramidChi2::sanitycheck_weak(): member value inconsistencer: numbinsinhistogram<=0");
		return (false);
	}
	
	if ((pyramidlevels!=NULL) && (numlevels<=0))
	{
		SG_ERROR("void CPyramidChi2::sanitycheck_weak(): inconsistency found: (pyramidlevels!=NULL) && (numlevels <=0)");
		
		return(false);
	}
	
	if ((pyramidlevels==NULL) && (numlevels>0))
	{
		SG_ERROR("void CPyramidChi2::sanitycheck_weak(): inconsistency found: (pyramidlevels==NULL) && (numlevels>0)");
		
		return(false);
	}
	
	if((weights!=NULL) &&(numweights<=0))
	{
		SG_ERROR("void CPyramidChi2::sanitycheck_weak(): inconsistency found: (weights!=NULL) && (numweights <=0)");
		
		return(false);
	}
	
	if ((weights==NULL) && (numweights>0))
	{
		SG_ERROR("void CPyramidChi2::sanitycheck_weak(): inconsistency found: (weights==NULL) && (numweights >0)");
		
		return(false);
	}
	

	INT sum=0;
	for (INT levelind=0; levelind < numlevels; ++levelind)
	{
		sum+=(ULONG) CMath::pow(4, pyramidlevels[levelind]);
	}
	
	if (sum!=numweights )
	{
		SG_ERROR("bool CPyramidChi2::sanitycheck_weak(): member value error: sum!=numweights ");
		return (false);
	}

	return (true);

}


DREAL CPyramidChi2::compute(INT idx_a, INT idx_b)
{
	// implied structure
	// for each level l in pyramidlevels we have at level l we have 4^l histograms with numbinsinhistogram bins
	//the features are a vector being a concatenation of histograms starting with all histograms at the largest level in pyramidlevels
	// then followed by all histograms at the next largest level in pyramidlevels, then the next largest and so on


	// the dimensionality is (LATEX) \sum_{ l \ in pyramidlevels } 4^l * numbinsinhistogram

	INT alen, blen;
	bool afree, bfree;

	DREAL* avec=
			((CRealFeatures*) lhs)->get_feature_vector(idx_a,
					alen, afree);
	DREAL* bvec=
			((CRealFeatures*) rhs)->get_feature_vector(idx_b,
					blen, bfree);
	ASSERT(alen==blen);

	INT dims=0;
	for (INT levelind=0; levelind <numlevels; ++levelind)
	{
		dims+=(INT)CMath::pow(4, pyramidlevels[levelind])*numbinsinhistogram;
	}
	ASSERT(dims==alen);

	//the actual computation - a weighted sum over chi2
	DREAL result=0;
	INT cursum=0;
	
	for (INT lvlind=0; lvlind< numlevels; ++lvlind)
	{
		for (INT histoind=0; histoind< (int)CMath::pow(4, pyramidlevels[lvlind]); ++histoind)
		{
			DREAL curweight=weights[cursum+histoind];
			
			for (INT i=0; i< numbinsinhistogram; ++i)
			{
				INT index= (cursum+histoind)*numbinsinhistogram+i;
				if(avec[index] + bvec[index]>0)
				{	
					result+= curweight*(avec[index] - bvec[index])*(avec[index]
						- bvec[index])/(avec[index] + bvec[index]);
				}
			}
		}
		cursum+=CMath::pow(4, pyramidlevels[lvlind]);
	}
	result=exp(-result/(DREAL)width);
	
	
	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return (result);
}


void CPyramidChi2::setstandardweights()
{
	INT sum=0;
	INT maxlvl=0;
	for (INT levelind=0; levelind < numlevels; ++levelind)
	{
		sum+=CMath::pow(4, pyramidlevels[levelind]);
		maxlvl=CMath::max(maxlvl,pyramidlevels[levelind]);
	}

	if (weights==NULL)
	{
		numweights=sum;
		weights=new DREAL[numweights];
	}

	else if (numweights!=sum)
	{
		// a possible source of error or leak!
		if (numweights>0)
		{
			delete[]  weights;
		}
		else
		{
			SG_ERROR("void CPyramidChi2::setstandardweights(): inconsistency found: (weights!=NULL) && (numweights <=0), continuing, but memory leak possible");
		}

		numweights=sum;
		weights=new DREAL[numweights];
	}
	//weights.resize(sum);
	
	INT cursum=0;
	for (INT levelind=0; levelind < numlevels; ++levelind)
	{
		if (pyramidlevels[levelind]==0)
		{
			for (INT histoind=0; histoind< (int)CMath::pow(4, pyramidlevels[levelind]); ++histoind)
			{
				weights[cursum+histoind]=CMath::pow((DREAL)2.0,
						-(DREAL)maxlvl);
			}
		}
		else
		{
			for (INT histoind=0; histoind< (INT)CMath::pow(4, pyramidlevels[levelind]); ++histoind)
			{
				weights[cursum+histoind]=CMath::pow((DREAL)2.0,
						(DREAL)(pyramidlevels[levelind]-1-maxlvl));
			}
		}
		cursum+=CMath::pow(4, pyramidlevels[levelind]);
	}
}
