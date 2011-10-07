/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/preprocessor/DiffusionMaps.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = i;

	CSimpleFeatures<double>* features = new CSimpleFeatures<double>(SGMatrix<double>(matrix,dim,N));
	SG_REF(features);
	CDiffusionMaps* dmaps = new CDiffusionMaps();
	dmaps->set_target_dim(2);
	dmaps->set_t(10);
	dmaps->parallel->set_num_threads(4);
	dmaps->apply_to_feature_matrix(features);
	SG_UNREF(dmaps);
	SG_UNREF(features);
	exit_shogun();
	return 0;
}
