/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/structure/HMSVMLabels.h>

using namespace shogun;

CHMSVMLabels::CHMSVMLabels()
: CStructuredLabels()
{
}

CHMSVMLabels::CHMSVMLabels(int32_t num_labels)
: CStructuredLabels(num_labels)
{
}

CHMSVMLabels::~CHMSVMLabels()
{
}

void CHMSVMLabels::add_label(SGVector< int32_t > label)
{
	CStructuredLabels::add_label( new CSequence(label) );
}
