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

#include "classifier/LPBoost.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/Cplex.h"

CLPBoost::CLPBoost() : CSparseLinearClassifier()
{
}


CLPBoost::~CLPBoost()
{
}

bool CLPBoost::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());
	INT num_train_labels=0;
	INT num_train_labels=get_labels()->get_num_labels();
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w=new DREAL[num_feat];
	w_dim=num_feat;
	ASSERT(w);

	CCplex solver;
	solver.init(LINEAR);
	//solver.setup_lpboost(get_features(), get_labels());

	while (true)
	{
		//add/remove constraints
		//check optimality
		solver.optimize(w, w_dim);
	}
	solver.cleanup();
	

	delete[] train_labels;

	return false;
}
#endif
