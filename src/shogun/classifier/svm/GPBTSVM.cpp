/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <stdio.h>
#include <shogun/classifier/svm/GPBTSVM.h>
#include <shogun/lib/external/gpdt.h>
#include <shogun/lib/external/gpdtsolve.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

CGPBTSVM::CGPBTSVM()
: CSVM(), model(NULL)
{
}

CGPBTSVM::CGPBTSVM(float64_t C, CKernel* k, CLabels* lab)
: CSVM(C, k, lab), model(NULL)
{
}

CGPBTSVM::~CGPBTSVM()
{
	SG_FREE(model);
}

bool CGPBTSVM::train_machine(CFeatures* data)
{
	float64_t* solution;                     /* store the solution found       */
	QPproblem prob;                          /* object containing the solvers  */

	ASSERT(kernel)
	ASSERT(m_labels && m_labels->get_num_labels())
	ASSERT(m_labels->get_label_type() == LT_BINARY)
	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		kernel->init(data, data);
	}

	SGVector<int32_t> lab=((CBinaryLabels*) m_labels)->get_int_labels();
	prob.KER=new sKernel(kernel, lab.vlen);
	prob.y=lab.vector;
	prob.ell=lab.vlen;
	SG_INFO("%d trainlabels\n", prob.ell)

	//  /*** set options defaults ***/
	prob.delta = epsilon;
	prob.maxmw = kernel->get_cache_size();
	prob.verbosity       = 0;
	prob.preprocess_size = -1;
	prob.projection_projector = -1;
	prob.c_const = get_C1();
	prob.chunk_size = get_qpsize();
	prob.linadd = get_linadd_enabled();

	if (prob.chunk_size < 2)      prob.chunk_size = 2;
	if (prob.q <= 0)              prob.q = prob.chunk_size / 3;
	if (prob.q < 2)               prob.q = 2;
	if (prob.q > prob.chunk_size) prob.q = prob.chunk_size;
	prob.q = prob.q & (~1);
	if (prob.maxmw < 5)
		prob.maxmw = 5;

	/*** set the problem description for final report ***/
	SG_INFO("\nTRAINING PARAMETERS:\n")
	SG_INFO("\tNumber of training documents: %d\n", prob.ell)
	SG_INFO("\tq: %d\n", prob.chunk_size)
	SG_INFO("\tn: %d\n", prob.q)
	SG_INFO("\tC: %lf\n", prob.c_const)
	SG_INFO("\tkernel type: %d\n", prob.ker_type)
	SG_INFO("\tcache size: %dMb\n", prob.maxmw)
	SG_INFO("\tStopping tolerance: %lf\n", prob.delta)

	//  /*** compute the number of cache rows up to maxmw Mb. ***/
	if (prob.preprocess_size == -1)
		prob.preprocess_size = (int32_t) ( (float64_t)prob.chunk_size * 1.5 );

	if (prob.projection_projector == -1)
	{
		if (prob.chunk_size <= 20) prob.projection_projector = 0;
		else prob.projection_projector = 1;
	}

	/*** compute the problem solution *******************************************/
	solution = SG_MALLOC(float64_t, prob.ell);
	prob.gpdtsolve(solution);
	/****************************************************************************/

	CSVM::set_objective(prob.objective_value);

	int32_t num_sv=0;
	int32_t bsv=0;
	int32_t i=0;
	int32_t k=0;

	for (i = 0; i < prob.ell; i++)
	{
		if (solution[i] > prob.DELTAsv)
		{
			num_sv++;
			if (solution[i] > (prob.c_const - prob.DELTAsv)) bsv++;
		}
	}

	create_new_model(num_sv);
	set_bias(prob.bee);

	SG_INFO("SV: %d BSV = %d\n", num_sv, bsv)

	for (i = 0; i < prob.ell; i++)
	{
		if (solution[i] > prob.DELTAsv)
		{
			set_support_vector(k, i);
			set_alpha(k++, solution[i]*((CBinaryLabels*) m_labels)->get_label(i));
		}
	}

	delete prob.KER;
	SG_FREE(solution);

	return true;
}
