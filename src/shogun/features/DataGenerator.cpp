/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/features/DataGenerator.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CDataGenerator::CDataGenerator() : CSGObject()
{
	init();
}

CDataGenerator::~CDataGenerator()
{

}

void CDataGenerator::init()
{
}

SGMatrix<float64_t> CDataGenerator::generate_mean_data(index_t m,
		index_t dim, float64_t mean_shift, float64_t* target_data)
{
	/* evtl use pre-allocated space */
	SGMatrix<float64_t> result;

	if (target_data)
	{
		result.matrix=target_data;
		result.num_rows=dim;
		result.num_cols=2*m;
	}
	else
		result=SGMatrix<float64_t>(dim, 2*m);

	/* fill matrix with normal data */
	for (index_t i=0; i<2*m; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			result(j,i)=CMath::randn_double();

		/* mean shift for second half */
		if (i>=m)
			result(0,i)+=mean_shift;
	}

	return result;
}
