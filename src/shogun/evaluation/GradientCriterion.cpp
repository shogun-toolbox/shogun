/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
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

