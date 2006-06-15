/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/KernelPerceptron.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CKernelPerceptron::CKernelPerceptron()
{
}


CKernelPerceptron::~CKernelPerceptron()
{
}

bool CKernelPerceptron::train()
{
	ASSERT(CKernelMachine::get_labels());
	//CLabels* train_labels=CKernelMachine::get_labels()->get_int_labels(num_train_labels);

//
//# compute output activation y = f(w x)
//# If y = t, don't change weights
//# If y != t, update the weights:
//
//w(new) = w(old) + 2 m t x
	return false;

}

bool CKernelPerceptron::load(FILE* srcfile)
{
	return false;
}

bool CKernelPerceptron::save(FILE* dstfile)
{
	return false;
}


DREAL CKernelPerceptron::classify_example(INT num)
{
	return 0;
}
