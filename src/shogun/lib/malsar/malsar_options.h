/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MALSAR_OPTIONS_H_
#define  MALSAR_OPTIONS_H_

#define IGNORE_IN_CLASSLIST

#include <stdlib.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
IGNORE_IN_CLASSLIST enum malsar_loss
{
	MALSAR_LOGISTIC,
	MALSAR_LEAST_SQUARES
};

IGNORE_IN_CLASSLIST struct malsar_options
{
	int termination;
	double tolerance;
	int max_iter;
	int n_tasks;
	int n_clusters;
	SGVector<int>* tasks_indices;
	malsar_loss loss;

	static malsar_options default_options()
	{
		malsar_options opts;
		opts.termination = 2;
		opts.tolerance = 1e-3;
		opts.max_iter = 1000;
		opts.tasks_indices = NULL;
		opts.n_clusters = 2;
		opts.loss = MALSAR_LOGISTIC;
		return opts;
	}
};

IGNORE_IN_CLASSLIST struct malsar_result_t
{
	SGMatrix<double> w;
	SGVector<double> c;

	malsar_result_t(SGMatrix<double> w_, SGVector<double> c_)
	{
		w = w_;
		c = c_;
	}
};
#endif
}
#endif   /* ----- #ifndef MALSAR_OPTIONS_H_  ----- */
