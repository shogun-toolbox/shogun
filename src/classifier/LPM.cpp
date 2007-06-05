/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef USE_CPLEX

#include "classifier/LPM.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/Cplex.h"

CLPM::CLPM() : CSparseLinearClassifier(), C1(1), C2(1), use_bias(true), epsilon(1e-3)
{
}


CLPM::~CLPM()
{
}

bool CLPM::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());
	INT num_train_labels=get_labels()->get_num_labels();
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w=new DREAL[num_feat];
	w_dim=num_feat;
	ASSERT(w);

	INT num_params=1+2*num_feat+num_vec; //b,w+,w-,xi
	DREAL* params=new DREAL[num_params];
	ASSERT(params);
	memset(params,0,sizeof(DREAL)*num_params);

	CCplex solver;
	solver.init(E_LINEAR);
	SG_INFO("C=%f\n", C1);
	solver.setup_lpm(C1, get_features(), get_labels(), get_bias_enabled());
	bool result=solver.optimize(params);
	solver.cleanup();

	set_bias(params[0]);
	for (INT i=0; i<num_feat; i++)
		w[i]=params[1+i]-params[1+num_feat+i];

//#define LPM_DEBUG
#ifdef LPM_DEBUG
	CMath::display_vector(params,num_params, "params");
	SG_PRINT("bias=%f\n", bias);
	CMath::display_vector(w,w_dim, "w");
	CMath::display_vector(&params[1],w_dim, "w+");
	CMath::display_vector(&params[1+w_dim],w_dim, "w-");
#endif
	delete[] params;
	return result;
}
#endif
