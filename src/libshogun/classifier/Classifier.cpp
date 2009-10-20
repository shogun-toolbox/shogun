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
BOOST_SERIALIZATION_ASSUME_ABSTRACT(shogun::CClassifier);
#endif //HAVE_BOOST_SERIALIZATION

using namespace shogun;

CClassifier::CClassifier() : CSGObject(), max_train_time(0), labels(NULL),
	solver_type(ST_AUTO)
{
}

CClassifier::~CClassifier()
{
    SG_UNREF(labels);
}
