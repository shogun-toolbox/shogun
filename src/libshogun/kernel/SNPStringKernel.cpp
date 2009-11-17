/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Berlin Institute of Technology
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/SNPStringKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

using namespace shogun;

CSNPStringKernel::CSNPStringKernel(int32_t size, int32_t degree)
: CStringKernel<char>(size), m_degree(degree)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CSNPStringKernel::CSNPStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t degree)
: CStringKernel<char>(10), m_degree(degree)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
}

CSNPStringKernel::~CSNPStringKernel()
{
	cleanup();
}

bool CSNPStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CSNPStringKernel::cleanup()
{
	CKernel::cleanup();
}

void CSNPStringKernel::obtain_base_strings()
{
	//should only be called on training data
	ASSERT(lhs==rhs);

	str_len=0;

	for (int32_t i=0; i<num_lhs; i++)
	{
		int32_t len;
		bool free_vec;
		char* vec = ((CStringFeatures<char>*) lhs)->get_feature_vector(i, len, free_vec);

		if (str_len==0)
		{
			str_len=len;
			size_t tlen=(len+1)*sizeof(char);
			str_min=(char*) malloc(tlen);
			str_maj=(char*) malloc(tlen);
			memset(str_min, 0, tlen);
			memset(str_maj, 0, tlen);
		}
		else
		{
			ASSERT(str_len==len);
		}

		for (int32_t j=0; j<len; j++)
		{
			// skip sequencing errors
			if (vec[j]=='0')
				continue;

			if (str_min[j]==0)
				str_min[j]=vec[j];
            else if (str_maj[j]==0 && vec[j]!=str_min[j])
				str_maj[j]=vec[j];
		}

		((CStringFeatures<char>*) lhs)->free_feature_vector(vec, i, free_vec);
	}

	for (int32_t j=0; j<str_len; j++)
	{
        // if only one one symbol occurs use 0
		if (str_min[j]==0)
            str_min[j]='0';
		if (str_maj[j]==0)
            str_maj[j]='0';

		if (str_min[j]>str_maj[j])
			CMath::swap(str_min[j], str_maj[j]);
	}
}

float64_t CSNPStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen);
	ASSERT(alen==str_len);
	ASSERT(str_min);
	ASSERT(str_maj);

	int32_t sumaa=0;
	int32_t sumbb=0;
	int32_t sumab=0;

	for (int32_t i = 0; i<alen-1; i+=2)
	{
		char a1=avec[i];
		char a2=avec[i+1];
		char b1=bvec[i];
		char b2=bvec[i+1];

		if (a1>a2)
			CMath::swap(a1, a2);
		if (b1>b2)
			CMath::swap(b1, b2);

		if ((a1!=a2 || a1=='0' || a1=='0') && (b1!=b2 || b1=='0' || b2=='0'))
			sumab++;
		else if (a1==a2 && b1==b2 && a1==b1)
		{
			if (a1==str_min[i])
				sumaa++;
			else if (a1==str_maj[i])
				sumbb++;
			else
            {
				SG_ERROR("The impossible happened i=%d a1=%c "
                        "a2=%c b1=%c b2=%c min=%c maj=%c\n", i, a1,a2, b1,b2, str_min[i], str_maj[i]);
            }
		}
	}

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return sumaa+sumbb+sumab;
}
