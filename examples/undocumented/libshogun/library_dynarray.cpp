/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/DynArray.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	DynArray<int32_t> values;

	for (int32_t i=0; i<1000; i++)
	{
		values.set_element(i,i);
	}

	for (int32_t i=0; i<1000; i++)
	{
		SG_SPRINT("values[%i]=%i\n", i, values[i]);
	}

	DynArray<SGVector<float64_t> > vectors(5);
	for (int32_t i=0; i<20; i++)
	{
		SG_SPRINT("%i\n", i);
		SGVector<float64_t> vec(i);

		for (int32_t j=0; j<i; j++)
			vec.vector[j]=j;

		vectors.set_element(vec,i);
	}

	for (int32_t i=0; i<20; i++)
	{
		SG_SPRINT("%i\n", i);
		vectors[i].display_vector();
	}

	exit_shogun();

	return 0;
}
