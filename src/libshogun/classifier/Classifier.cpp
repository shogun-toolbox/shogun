/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/Classifier.h"


#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>
BOOST_IS_ABSTRACT(CClassifier);
#endif //HAVE_BOOST_SERIALIZATION


CClassifier::CClassifier() : CSGObject(), max_train_time(0), labels(NULL),
	solver_type(ST_AUTO)
{
}

CClassifier::~CClassifier()
{
    SG_UNREF(labels);
}

CLabels* CClassifier::classify(CLabels* output)
{
	if (labels)
	{
		int32_t num=labels->get_num_labels();
		ASSERT(num>0);

		if (!output)
		{
			output=new CLabels(num);
			SG_REF(output);
		}

		for (int32_t i=0; i<num; i++)
			output->set_label(i, classify_example(i));

		return output;
	}

	return NULL;
}
