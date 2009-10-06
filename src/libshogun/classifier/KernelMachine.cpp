/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/KernelMachine.h"


#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>
BOOST_IS_ABSTRACT(CKernelMachine);
#endif //HAVE_BOOST_SERIALIZATION


CKernelMachine::CKernelMachine()
: CClassifier(), kernel(NULL), use_batch_computation(true), use_linadd(true)
{
}

CKernelMachine::~CKernelMachine()
{
	SG_UNREF(kernel);
}

CLabels* CKernelMachine::classify(CLabels* output)
{
	if (kernel && kernel->has_features())
	{
		int32_t num=kernel->get_num_vec_rhs();
		ASSERT(num>0);

		if (!output)
		{
			output=new CLabels(num);
			SG_REF(output);
		}
		ASSERT(output->get_num_labels()==num);

		for (int32_t i=0; i<num; i++)
			output->set_label(i, classify_example(i));

		return output;
	}

	return NULL;
}

CLabels* CKernelMachine::classify(CFeatures* data)
{
	if (!kernel)
		SG_ERROR("No kernel assigned!\n");

	CFeatures* lhs=kernel->get_lhs();
	if (!lhs || !lhs->get_num_vectors())
	{
		SG_UNREF(lhs);
		SG_ERROR("No vectors on left hand side\n");
	}
	kernel->init(lhs, data);
	SG_UNREF(lhs);

	return classify((CLabels*) NULL);
}
