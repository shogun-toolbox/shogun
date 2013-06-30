/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/classifier/GaussianProcessBinaryClassification.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianProcessBinaryClassification::CGaussianProcessBinaryClassification()
	: CGaussianProcessMachine()
{
}

CGaussianProcessBinaryClassification::CGaussianProcessBinaryClassification(CInferenceMethod* method)
	: CGaussianProcessMachine(method)
{
	// set labels
	m_labels=method->get_labels();
}

CGaussianProcessBinaryClassification::~CGaussianProcessBinaryClassification()
{
}

CBinaryLabels* CGaussianProcessBinaryClassification::apply_binary(CFeatures* data)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CGaussianProcessBinaryClassification::train_machine(CFeatures* data)
{
	SG_NOTIMPLEMENTED
	return false;
}

#endif /* HAVE_EIGEN3 */
