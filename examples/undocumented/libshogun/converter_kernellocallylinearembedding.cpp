/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/KernelLocallyLinearEmbedding.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = CMath::sin((i/float64_t(N*dim))*3.14);

	CDenseFeatures<double>* features = new CDenseFeatures<double>(SGMatrix<double>(matrix,dim,N));
	SG_REF(features);
	CKernelLocallyLinearEmbedding* klle = new CKernelLocallyLinearEmbedding();
	CKernel* kernel = new CLinearKernel();
	klle->set_target_dim(2);
	klle->set_k(4);
	klle->set_kernel(kernel);
	klle->parallel->set_num_threads(4);
	CDenseFeatures<double>* embedding = klle->embed(features);
	SG_UNREF(embedding);
	SG_UNREF(klle);
	SG_UNREF(features);
	exit_shogun();
	return 0;
}
