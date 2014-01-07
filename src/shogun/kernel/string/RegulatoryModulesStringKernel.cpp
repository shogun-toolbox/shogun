/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Sebastian J. Schultheiss and Soeren Sonnenburg
 * Copyright (C) 2009 Max-Planck-Society
 */

#include <lib/common.h>
#include <kernel/string/RegulatoryModulesStringKernel.h>
#include <features/Features.h>
#include <io/SGIO.h>

using namespace shogun;

CRegulatoryModulesStringKernel::CRegulatoryModulesStringKernel()
: CStringKernel<char>(0)
{
	init();
}

CRegulatoryModulesStringKernel::CRegulatoryModulesStringKernel(
		int32_t size, float64_t w, int32_t d, int32_t s, int32_t wl)
: CStringKernel<char>(size)
{
	init();
}

CRegulatoryModulesStringKernel::CRegulatoryModulesStringKernel(CStringFeatures<char>* lstr, CStringFeatures<char>* rstr,
		CDenseFeatures<uint16_t>* lpos, CDenseFeatures<uint16_t>* rpos,
		float64_t w, int32_t d, int32_t s, int32_t wl, int32_t size)
: CStringKernel<char>(size)
{
	init();
	set_motif_positions(lpos, rpos);
	init(lstr,rstr);
}

CRegulatoryModulesStringKernel::~CRegulatoryModulesStringKernel()
{
	SG_UNREF(motif_positions_lhs);
	SG_UNREF(motif_positions_rhs);
}

void CRegulatoryModulesStringKernel::init()
{
	width=0;
	degree=0;
	shift=0;
	window=0;
	motif_positions_lhs=NULL;
	motif_positions_rhs=NULL;

	SG_ADD(&width, "width", "the width of Gaussian kernel part", MS_AVAILABLE);
	SG_ADD(&degree, "degree", "the degree of weighted degree kernel part",
	    MS_AVAILABLE);
	SG_ADD(&shift, "shift",
	    "the shift of weighted degree with shifts kernel part", MS_AVAILABLE);
	SG_ADD(&window, "window", "the size of window around motifs", MS_AVAILABLE);
	SG_ADD((CSGObject**)&motif_positions_lhs, "motif_positions_lhs",
			"the matrix of motif positions from sequences left-hand side", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&motif_positions_rhs, "motif_positions_rhs",
			"the matrix of motif positions from sequences right-hand side", MS_NOT_AVAILABLE);
	SG_ADD(&position_weights, "position_weights", "scaling weights in window", MS_NOT_AVAILABLE);
	SG_ADD(&weights, "weights", "weights of WD kernel", MS_NOT_AVAILABLE);
}

bool CRegulatoryModulesStringKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(motif_positions_lhs)
	ASSERT(motif_positions_rhs)

	if (l->get_num_vectors() != motif_positions_lhs->get_num_vectors())
		SG_ERROR("Number of vectors does not agree (LHS: %d, Motif LHS: %d).\n",
				l->get_num_vectors(),  motif_positions_lhs->get_num_vectors());
	if (r->get_num_vectors() != motif_positions_rhs->get_num_vectors())
		SG_ERROR("Number of vectors does not agree (RHS: %d, Motif RHS: %d).\n",
				r->get_num_vectors(), motif_positions_rhs->get_num_vectors());

	set_wd_weights();
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CRegulatoryModulesStringKernel::set_motif_positions(
		CDenseFeatures<uint16_t>* positions_lhs, CDenseFeatures<uint16_t>* positions_rhs)
{
	ASSERT(positions_lhs)
	ASSERT(positions_rhs)
	SG_UNREF(motif_positions_lhs);
	SG_UNREF(motif_positions_rhs);
	if (positions_lhs->get_num_features() != positions_rhs->get_num_features())
		SG_ERROR("Number of dimensions does not agree.\n")

	motif_positions_lhs=positions_lhs;
	motif_positions_rhs=positions_rhs;
	SG_REF(positions_lhs);
	SG_REF(positions_rhs);
}

float64_t CRegulatoryModulesStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	ASSERT(motif_positions_lhs)
	ASSERT(motif_positions_rhs)

	bool free_avec, free_bvec;
	int32_t alen=0;
	int32_t blen=0;
	char* avec=((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec=((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	int32_t alen_pos, blen_pos;
	bool afree_pos, bfree_pos;
	uint16_t* positions_a = motif_positions_lhs->get_feature_vector(idx_a, alen_pos, afree_pos);
	uint16_t* positions_b = motif_positions_rhs->get_feature_vector(idx_b, blen_pos, bfree_pos);
	ASSERT(alen_pos==blen_pos)
	int32_t num_pos=alen_pos;


	float64_t result_rbf=0;
	float64_t result_wds=0;

	for (int32_t p=0; p<num_pos; p++)
	{
		result_rbf+=CMath::sq(positions_a[p]-positions_b[p]);

		for (int32_t p2=0; p2<num_pos; p2++) //p+1 and below * 2
			result_rbf+=CMath::sq( (positions_a[p]-positions_a[p2]) - (positions_b[p]-positions_b[p2]) );

		int32_t limit = window;
		if (window + positions_a[p] > alen)
			limit = alen - positions_a[p];

		if (window + positions_b[p] > blen)
			limit = CMath::min(limit, blen - positions_b[p]);

		result_wds+=compute_wds(&avec[positions_a[p]], &bvec[positions_b[p]],
				limit);
	}

	float64_t result=exp(-result_rbf/width)+result_wds;

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	((CDenseFeatures<uint16_t>*) lhs)->free_feature_vector(positions_a, idx_a, afree_pos);
	((CDenseFeatures<uint16_t>*) rhs)->free_feature_vector(positions_b, idx_b, bfree_pos);

	return result;
}

float64_t CRegulatoryModulesStringKernel::compute_wds(
	char* avec, char* bvec, int32_t len)
{
	float64_t* max_shift_vec = SG_MALLOC(float64_t, shift);
	float64_t sum0=0 ;
	for (int32_t i=0; i<shift; i++)
		max_shift_vec[i]=0 ;

	// no shift
	for (int32_t i=0; i<len; i++)
	{
		if ((position_weights.vector!=NULL) && (position_weights[i]==0.0))
			continue ;

		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<len); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights.vector!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;

	for (int32_t i=0; i<len; i++)
	{
		for (int32_t k=1; (k<=shift) && (i+k<len); k++)
		{
			if ((position_weights.vector!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			for (int32_t j=0; (j<degree) && (i+j+k<len); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			for (int32_t j=0; (j<degree) && (i+j+k<len); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[j];
			}
			if (position_weights.vector!=NULL)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
	}

	float64_t result = sum0 ;
	for (int32_t i=0; i<shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	SG_FREE(max_shift_vec);
	return result ;
}

void CRegulatoryModulesStringKernel::set_wd_weights()
{
	ASSERT(degree>0)

	weights=SGVector<float64_t>(degree);

	int32_t i;
	float64_t sum=0;
	for (i=0; i<degree; i++)
	{
		weights[i]=degree-i;
		sum+=weights[i];
	}

	for (i=0; i<degree; i++)
		weights[i]/=sum;
}

