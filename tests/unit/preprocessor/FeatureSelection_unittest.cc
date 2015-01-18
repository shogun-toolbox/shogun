/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/statistics/HSIC.h>
#include <shogun/preprocessor/BAHSIC.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(FeatureSelection, remove_feats)
{
	const index_t dim=8;
	const index_t num_data=5;

	// use fix seed for reproducibility
	CMath::init_random(1);

	SGMatrix<float64_t> data(dim, num_data);
	for (index_t i=0; i<dim*num_data; ++i)
		data.matrix[i]=i;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	CBAHSIC* fs=new CBAHSIC();
	index_t target_dim=dim/2;
	fs->set_num_remove(dim-target_dim);
	fs->set_policy(N_LARGEST);

	// create a dummy argsorted vector to remove last dim/2 features
	SGVector<index_t> argsorted(dim);
	argsorted.range_fill();

	CFeatures* reduced=fs->remove_feats(feats, argsorted);
	SGMatrix<float64_t> reduced_data
		=((CDenseFeatures<float64_t>*)reduced)->get_feature_matrix();

	for (index_t i=0; i<target_dim; ++i)
	{
		for (index_t j=0; j<num_data; ++j)
			EXPECT_NEAR(data(i, j), reduced_data(i, j), 1E-15);
	}

	SG_UNREF(reduced);
	SG_UNREF(fs);
}

TEST(FeatureSelection, compute_measures)
{
	const index_t dim=8;
	const index_t num_data=5;

	// use fix seed for reproducibility
	CMath::init_random(1);

	SGMatrix<float64_t> data(dim, num_data);
	for (index_t i=0; i<dim*num_data; ++i)
		data.matrix[i]=(i+1.0)/dim/num_data;

	SGVector<float64_t> labels_vec(num_data);
	for (index_t i=0; i<num_data; ++i)
		labels_vec[i]=CMath::random(0, 1);

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);
	CBinaryLabels* labels=new CBinaryLabels(labels_vec);
	float64_t sigma=1.0;
	CGaussianKernel* kernel_p=new CGaussianKernel(10, 2*CMath::sq(sigma));
	CGaussianKernel* kernel_q=new CGaussianKernel(10, 2*CMath::sq(sigma));

	// SG_REF'ing the kernel for q because it is SG_UNREF'ed in precompute
	// call and to replace by a CCustomKernel
	SG_REF(kernel_q);

	CBAHSIC* fs=new CBAHSIC();
	fs->set_labels(labels);
	fs->set_kernel_features(kernel_p);
	fs->set_kernel_labels(kernel_q);

	// compute the measure removing dimension 0
	float64_t measure=fs->compute_measures(feats, 0);

	// recreate this using HSIC
	SGVector<index_t> inds(dim-1);
	for (index_t i=0; i<inds.vlen; ++i)
		inds[i]=i+1;
	CFeatures* transformed=feats->copy_dimension_subset(inds);

	SGMatrix<float64_t> l_data(1, num_data);
	memcpy(l_data.matrix, labels_vec.vector, sizeof(float64_t)*num_data);
	CDenseFeatures<float64_t>* l_feats=new CDenseFeatures<float64_t>(l_data);

	CHSIC* hsic=new CHSIC();
	hsic->set_p(transformed);
	hsic->set_q(l_feats);
	hsic->set_kernel_p(kernel_p);
	hsic->set_kernel_q(kernel_q);

	EXPECT_NEAR(measure, hsic->compute_statistic(), 1E-15);

	SG_UNREF(fs);
	SG_UNREF(hsic);
	SG_UNREF(kernel_q);
	SG_UNREF(feats);
	SG_UNREF(transformed);
}
