/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <base/init.h>
#include <features/DenseFeatures.h>
#include <converter/LocallyLinearEmbedding.h>
#include <mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = CMath::sin((i/float64_t(N*dim))*3.14);

	CDenseFeatures<double>* features = new CDenseFeatures<double>(SGMatrix<double>(matrix,dim,N));
	SG_REF(features);
	CLocallyLinearEmbedding* lle = new CLocallyLinearEmbedding();
	lle->set_target_dim(2);
	lle->set_k(4);
	lle->parallel->set_num_threads(4);
	CDenseFeatures<double>* embedding = lle->embed(features);
	SG_UNREF(embedding);
	SG_UNREF(lle);
	SG_UNREF(features);
	exit_shogun();
	return 0;
}
#else
int main(int argc, char **argv)
{
	return 0;
}
#endif
