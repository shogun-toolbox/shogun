/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "GNB.h"
#include "Classifier.h"
#include "features/Features.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/Signal.h"

using namespace shogun;

CGNB::CGNB() : CClassifier(), num_train_labels(0), num_classes(0)
{

};

CGNB::CGNB(CFeatures* train_examples, CLabels* train_labels) : CClassifier()
{
	ASSERT(train_examples->get_num_vectors() == train_labels->get_num_labels());
	num_train_labels = train_labels->get_num_labels();
	set_labels(train_labels);
};

CGNB::~CGNB()
{
	delete labels;
};

bool CGNB::train(CFeatures* data)
{
	ASSERT(data->get_num_vectors() == num_train_labels);
	SG_NOTIMPLEMENTED;
	return NULL;
}

CLabels* CGNB::classify()
{
	SG_NOTIMPLEMENTED;
	return NULL;
};

CLabels* CGNB::classify(CFeatures* data)
{
	SG_NOTIMPLEMENTED;
	return NULL;
};

float64_t CGNB::classify_example(int32_t idx)
{
	SG_NOTIMPLEMENTED;
	return 0.0;
};
