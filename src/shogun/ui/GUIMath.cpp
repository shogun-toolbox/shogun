/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <ui/GUIMath.h>
#include <ui/SGInterface.h>

#include <lib/config.h>
#include <io/SGIO.h>
#include <mathematics/Math.h>

using namespace shogun;

CGUIMath::CGUIMath(CSGInterface* ui_)
: CSGObject(), ui(ui_), threshold(0.0)
{
}

void CGUIMath::set_threshold(float64_t value)
{
	SG_INFO("Old threshold: %f.\n", threshold)
	threshold=value;
	SG_INFO("New threshold: %f.\n", threshold)
}

void CGUIMath::init_random(uint32_t initseed)
{
	CMath::init_random(initseed);
}
