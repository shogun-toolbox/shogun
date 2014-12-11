/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifdef USE_CPLEX

#include <stdio.h>
#include <shogun/classifier/LPBoost.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Cplex.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>

using namespace shogun;

CLPBoost::CLPBoost()
: CLinearMachine(), C1(1), C2(1), use_bias(true), epsilon(1e-3)
{
	u=NULL;
	dim=NULL;
	num_sfeat=0;
	num_svec=0;
	sfeat=NULL;
}


CLPBoost::~CLPBoost()
{
	cleanup();
}

bool CLPBoost::init(int32_t num_vec)
{
	u=SG_MALLOC(float64_t, num_vec);
	for (int32_t i=0; i<num_vec; i++)
		u[i]=1.0/num_vec;

	dim=new CDynamicArray<int32_t>(100000);

	sfeat= ((CSparseFeatures<float64_t>*) features)->get_transposed(num_sfeat, num_svec);

	if (sfeat)
		return true;
	else
		return false;
}

void CLPBoost::cleanup()
{
	SG_FREE(u);
	u=NULL;

	((CSparseFeatures<float64_t>*) features)->clean_tsparse(sfeat, num_svec);
	sfeat=NULL;

	delete dim;
	dim=NULL;
}

float64_t CLPBoost::find_max_violator(int32_t& max_dim)
{
	float64_t max_val=0;
	max_dim=-1;

	for (int32_t i=0; i<num_svec; i++)
	{
		float64_t valplus=0;
		float64_t valminus=0;

		for (int32_t j=0; j<sfeat[i].num_feat_entries; j++)
		{
			int32_t idx=sfeat[i].features[j].feat_index;
			float64_t v=u[idx]*((CBinaryLabels*)m_labels)->get_confidence(idx)*sfeat[i].features[j].entry;
			valplus+=v;
			valminus-=v;
		}

		if (valplus>max_val || max_dim==-1)
		{
			max_dim=i;
			max_val=valplus;
		}

		if (valminus>max_val)
		{
			max_dim=num_svec+i;
			max_val=valminus;
		}
	}

	dim->append_element(max_dim);
	return max_val;
}

bool CLPBoost::train_machine(CFeatures* data)
{
	ASSERT(m_labels)
	ASSERT(features)
	int32_t num_train_labels=m_labels->get_num_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels)
	w = SGVector<float64_t>(num_feat);
	memset(w.vector,0,sizeof(float64_t)*num_feat);

	CCplex solver;
	solver.init(E_LINEAR);
	SG_PRINT("setting up lpboost\n")
	solver.setup_lpboost(C1, num_vec);
	SG_PRINT("finished setting up lpboost\n")

	float64_t result=init(num_vec);
	ASSERT(result)

	int32_t num_hypothesis=0;
	CTime time;
	CSignal::clear_cancel();

	while (!(CSignal::cancel_computations()))
	{
		int32_t max_dim=0;
		float64_t violator=find_max_violator(max_dim);
		SG_PRINT("iteration:%06d violator: %10.17f (>1.0) chosen: %d\n", num_hypothesis, violator, max_dim)
		if (violator <= 1.0+epsilon && num_hypothesis>1) //no constraint violated
		{
			SG_PRINT("converged after %d iterations!\n", num_hypothesis)
			break;
		}

		float64_t factor=+1.0;
		if (max_dim>=num_svec)
		{
			factor=-1.0;
			max_dim-=num_svec;
		}

		SGSparseVectorEntry<float64_t>* h=sfeat[max_dim].features;
		int32_t len=sfeat[max_dim].num_feat_entries;
		solver.add_lpboost_constraint(factor, h, len, num_vec, m_labels);
		solver.optimize(u);
		//CMath::display_vector(u, num_vec, "u");
		num_hypothesis++;

		if (get_max_train_time()>0 && time.cur_time_diff()>get_max_train_time())
			break;
	}
	float64_t* lambda=SG_MALLOC(float64_t, num_hypothesis);
	solver.optimize(u, lambda);

	//CMath::display_vector(lambda, num_hypothesis, "lambda");
	for (int32_t i=0; i<num_hypothesis; i++)
	{
		int32_t d=dim->get_element(i);
		if (d>=num_svec)
			w[d-num_svec]+=lambda[i];
		else
			w[d]-=lambda[i];

	}
	//solver.write_problem("problem.lp");
	solver.cleanup();

	cleanup();

	return true;
}
#endif
