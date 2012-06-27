/*
 * GradientCriterion.cpp
 *
 *  Created on: Jun 27, 2012
 *      Author: jacobw
 */

#include <shogun/evaluation/GradientCriterion.h>

using namespace shogun;

CGradientCriterion::CGradientCriterion() : CEvaluation()
{
	m_direction = ED_MINIMIZE;
}

CGradientCriterion::~CGradientCriterion()
{
}

