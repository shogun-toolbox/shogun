/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  SLEP_OPTIONS_H_
#define  SLEP_OPTIONS_H_

#define IGNORE_IN_CLASSLIST

#include <stdlib.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
IGNORE_IN_CLASSLIST enum slep_mode
{
	MULTITASK_GROUP,
	MULTITASK_TREE,
	FEATURE_GROUP,
	FEATURE_TREE
};

IGNORE_IN_CLASSLIST struct slep_options
{
	bool general;
	int termination;
	double tolerance;
	int max_iter;
	int restart_num;
	int n_nodes;
	int n_tasks;
	int regularization;
	int n_feature_blocks;
	int* ind;
	double* ind_t;
	double* G;
	double* gWeight;
	double q;
	slep_mode mode;

	static slep_options default_options()
	{
		slep_options opts;
		opts.general = false;
		opts.termination = 2;
		opts.tolerance = 1e-3;
		opts.max_iter = 1000;
		opts.restart_num = 100;
		opts.regularization = 0;
		opts.q = 2.0;
		opts.gWeight = NULL;
		opts.ind = NULL;
		opts.ind_t = NULL;
		opts.G = NULL;
		opts.mode = MULTITASK_GROUP;
		return opts;
	}
};

IGNORE_IN_CLASSLIST struct slep_result_t
{
	SGMatrix<double> w;
	SGVector<double> c;

	slep_result_t(SGMatrix<double> w_, SGVector<double> c_)
	{
		w = w_;
		c = c_;
	}
};
#endif
}
#endif   /* ----- #ifndef SLEP_OPTIONS_H_  ----- */


