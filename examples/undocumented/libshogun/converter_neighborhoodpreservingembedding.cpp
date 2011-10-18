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
#include <shogun/converter/NeighborhoodPreservingEmbedding.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = CMath::random(1,100);

	CSimpleFeatures<double>* features = new CSimpleFeatures<double>(SGMatrix<double>(matrix,dim,N));
	SG_REF(features);
	CNeighborhoodPreservingEmbedding* npe = new CNeighborhoodPreservingEmbedding();
	npe->set_target_dim(2);
	npe->set_k(15);
	npe->parallel->set_num_threads(4);
	CSimpleFeatures<double>* embedding = npe->embed(features);
	SG_UNREF(embedding);
	SG_UNREF(npe);
	SG_UNREF(features);
	exit_shogun();
	return 0;
}
