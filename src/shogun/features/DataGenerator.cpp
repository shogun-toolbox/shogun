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
		index_t dim, float64_t mean_shift)
{
	SGMatrix<float64_t> result(dim, m);

	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			result(j,i)=CMath::randn_double();

		result(0,i)+=mean_shift;
	}

	return result;
}
