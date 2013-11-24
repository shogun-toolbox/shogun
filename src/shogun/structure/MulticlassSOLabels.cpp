/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/structure/MulticlassSOLabels.h>

using namespace shogun;

CMulticlassSOLabels::CMulticlassSOLabels()
: CStructuredLabels()
{
	init();
}

CMulticlassSOLabels::CMulticlassSOLabels(SGVector< float64_t > const src)
: CStructuredLabels(src.vlen)
{
	init();

	m_num_classes = SGVector< float64_t >::max(src.vector, src.vlen) + 1;
	for ( int32_t i = 0 ; i < src.vlen ; ++i )
	{
		if ( src[i] < 0 || src[i] >= m_num_classes )
			SG_ERROR("Found label out of {0, 1, 2, ..., num_classes-1}")
		else
			add_label( new RealNumber(src[i]) );
	}

	//TODO check that every class has at least one example
}

CMulticlassSOLabels::~CMulticlassSOLabels()
{
}

void CMulticlassSOLabels::init()
{
	SG_ADD(&m_num_classes, "m_num_classes", "The number of classes",
			MS_NOT_AVAILABLE);

	m_num_classes = 0;
}
