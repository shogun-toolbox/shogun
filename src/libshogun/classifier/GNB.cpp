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

CGNB::CGNB() : CClassifier()
{

};

CGNB::CGNB(CFeatures* train_examples, CLabels* train_labels) : CClassifier()
{
	ASSERT(train_examples->get_num_vectors() == train_labels->get_num_labels());
};

CGNB::~CGNB()
{

};

bool CGNB::train(CFeatures* data)
{
	SG_NOTIMPLEMENTED;
	return;
}

CLabels* CGNB::classify()
{
	SG_NOTIMPLEMENTED;
	return;
};

CLabels* CGNB::classify(CFeatures* data)
{
	SG_NOTIMPLEMENTED;
	return;
};

float64_t CGNB::classify_example(int32_t idx)
{
	SG_NOTIMPLEMENTED;
	return;
};

bool CGNB::save(FILE* dstfile)
{
	SG_NOTIMPLEMENTED;
	return;
};

bool CGNB::load(FILE* srcfile)
{
	SG_NOTIMPLEMENTED;
	return;
};
