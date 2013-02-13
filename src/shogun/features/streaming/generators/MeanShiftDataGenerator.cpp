/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/lib/common.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>

using namespace shogun;

CMeanShiftDataGenerator::CMeanShiftDataGenerator() :
		CStreamingDenseFeatures<float64_t>()
{
	init();
}

CMeanShiftDataGenerator::CMeanShiftDataGenerator(float64_t mean_shift,
		index_t dimension): CStreamingDenseFeatures<float64_t>()
{
	init();
	set_mean_shift_model(mean_shift, dimension);
}

CMeanShiftDataGenerator::~CMeanShiftDataGenerator()
{
}

void CMeanShiftDataGenerator::set_mean_shift_model(float64_t mean_shift,
		index_t dimension)
{
	m_dimension=dimension;
	m_mean_shift=mean_shift;
}

void CMeanShiftDataGenerator::init()
{
	m_dimension=0;
	m_mean_shift=0;
}

bool CMeanShiftDataGenerator::get_next_example()
{
	SG_SDEBUG("entering CMeanShiftDataGenerator::get_next_example()\n");

	/* allocate space */
	SGVector<float64_t> result=SGVector<float64_t>(m_dimension);

	/* fill with std normal data */
	for (index_t i=0; i<m_dimension; ++i)
		result[i]=CMath::randn_double();

	/* mean shift in first dimension */
	result[0]+=m_mean_shift;

	/* save example back to superclass */
	CMeanShiftDataGenerator::current_vector=result;

	SG_SDEBUG("leaving CMeanShiftDataGenerator::get_next_example()\n");
	return true;
}

void CMeanShiftDataGenerator::release_example()
{
	SGVector<float64_t> temp=SGVector<float64_t>();
	CMeanShiftDataGenerator::current_vector=temp;
}
