/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Alexander Binder
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/kernel/PyramidChi2.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CPyramidChi2::CPyramidChi2(void)
: weights(NULL)
{
	// this will produce an erro in kernel computation!
	num_cells=0;
	width_computation_type=0;
	width=1;
	num_randfeats_forwidthcomputation=-1;
}

CPyramidChi2::CPyramidChi2(
	int32_t size, int32_t num_cells2,
		float64_t* weights_foreach_cell2, 
		int32_t width_computation_type2,
		float64_t width2)
: CDotKernel(size), num_cells(num_cells2),weights(NULL),
width_computation_type(width_computation_type2), width(width2),
	 num_randfeats_forwidthcomputation(-1)
{
	if(num_cells<=0)
		SG_ERROR("CPyramidChi2 Constructor fatal error: parameter num_cells2 NOT positive");
	weights=new float64_t[num_cells];
	if(weights_foreach_cell2)
	{
		for (int32_t i=0; i<num_cells; ++i)
			weights[i]=weights_foreach_cell2[i];
	}
	else
	{	for (int32_t i=0; i<num_cells; ++i)
			weights[i]=1;
	}

	if (width_computation_type>0 )
	{
		num_randfeats_forwidthcomputation=(int32_t)CMath::round(width);
		width=-1;
	}

	
}

void CPyramidChi2::cleanup()
{
	// this will produce an erro in kernel computation!
	num_cells=0;
	width_computation_type=0;
	width=1;

	num_randfeats_forwidthcomputation=-1;

	delete[] weights;
	weights=NULL;

	CKernel::cleanup();
}

bool CPyramidChi2::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	return init_normalizer();
}

CPyramidChi2::CPyramidChi2(
	CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r, 
		int32_t size, int32_t num_cells2,
		float64_t* weights_foreach_cell2, 
		int32_t width_computation_type2,
		float64_t width2)
: CDotKernel(size), num_cells(num_cells2), weights(NULL),
width_computation_type(width_computation_type2), width(width2),
	  num_randfeats_forwidthcomputation(-1)
{
	if(num_cells<=0)
		SG_ERROR("CPyramidChi2 Constructor fatal error: parameter num_cells2 NOT positive");
	weights=new float64_t[num_cells];
	if(weights_foreach_cell2)
	{
		for (int32_t i=0; i<num_cells; ++i)
			weights[i]=weights_foreach_cell2[i];
	}
	else
	{	for (int32_t i=0; i<num_cells; ++i)
			weights[i]=1;
	}

	if (width_computation_type>0 )
	{
		num_randfeats_forwidthcomputation=(int32_t)CMath::round(width);
		width=-1;
	}

	init(l, r);
}

CPyramidChi2::~CPyramidChi2()
{
	cleanup();
}



float64_t CPyramidChi2::compute(int32_t idx_a, int32_t idx_b)
{

	if(num_cells<=0)
		SG_ERROR("CPyramidChi2::compute(...) fatal error: parameter num_cells NOT positive");

	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a,
					alen, afree);
	float64_t* bvec=((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b,
					blen, bfree);
	if(alen!=blen)
		SG_ERROR("CPyramidChi2::compute(...) fatal error: lhs feature dim != rhs feature dim");	

	int32_t dims=alen/num_cells;


	if(width<=0)
	{
		if(width_computation_type >0)
		{
			
			//compute width
			int32_t numind;

			if (num_randfeats_forwidthcomputation >1)
			{
				numind=CMath::min( ((CSimpleFeatures<float64_t>*) lhs)->get_num_vectors() , num_randfeats_forwidthcomputation);
			}
			else
			{
				numind= ((CSimpleFeatures<float64_t>*) lhs)->get_num_vectors();
			}
			float64_t* featindices = new float64_t[numind];

			if (num_randfeats_forwidthcomputation >0)
			{
				for(int32_t i=0; i< numind;++i)
					featindices[i]=CMath::random(0, ((CSimpleFeatures<float64_t>*) lhs)->get_num_vectors()-1);
			}
			else
			{
				for(int32_t i=0; i< numind;++i)
					featindices[i]=i;
			}
			

			width=0;

			//get avec, get bvec	only from lhs, do not free	
			for (int32_t li=0; li < numind;++li)
			{
				avec=((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(featindices[li],
					alen, afree);
				for (int32_t ri=0; ri <=li;++ri)
				{
					// lhs is right here!!!
					bvec=((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(featindices[ri],
							blen, bfree);
				
					float64_t result=0;
					for (int32_t histoind=0; histoind<num_cells; ++histoind)
					{
						float64_t curweight=weights[histoind];
			
						for (int32_t i=0; i< dims; ++i)
						{
							int32_t index= histoind*dims+i;
							if(avec[index] + bvec[index]>0)
							{	
								result+= curweight*(avec[index] - bvec[index])*(avec[index]
									- bvec[index])/(avec[index] + bvec[index]);
							}
						}
					}
					width+=result*2.0/((double)numind)/(numind+1.0);
				}

			}
			delete[] featindices;
		}
		else
		{
			SG_ERROR("CPyramidChi2::compute(...) fatal error: width<=0");
		}
	}


	//the actual kernel computation
	avec=((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a,
					alen, afree);
	bvec=((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b,
					blen, bfree);

	float64_t result=0;
	for (int32_t histoind=0; histoind<num_cells; ++histoind)
	{
		float64_t curweight=weights[histoind];
			
		for (int32_t i=0; i< dims; ++i)
		{
			int32_t index= histoind*dims+i;
			if(avec[index] + bvec[index]>0)
			{	
				result+= curweight*(avec[index] - bvec[index])*(avec[index]
					- bvec[index])/(avec[index] + bvec[index]);
			}
		}
	}
	result= CMath::exp(-result/width);
	
	
	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return (result);
}

void CPyramidChi2::setparams_pychi2(int32_t num_cells2,
		float64_t* weights_foreach_cell2, 
		int32_t width_computation_type2,
		float64_t width2)
{
	num_cells=num_cells2;
	width_computation_type=width_computation_type2; 
	width=width2;
	num_randfeats_forwidthcomputation=-1;

	if(num_cells<=0)
		SG_ERROR("CPyramidChi2::setparams_pychi2(...) fatal error: parameter num_cells2 NOT positive");
	if(weights)
		delete[] weights;
	weights=new float64_t[num_cells];
	if(weights_foreach_cell2)
	{
		for (int32_t i=0; i<num_cells; ++i)
			weights[i]=weights_foreach_cell2[i];
	}
	else
	{	for (int32_t i=0; i<num_cells; ++i)
			weights[i]=1;
	}

	if (width_computation_type>0 )
	{
		num_randfeats_forwidthcomputation=(int32_t)CMath::round(width);
		width=-1;
	}
}
