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
#include <shogun/preprocessor/BAHSIC.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(BAHSIC, apply)
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

	CBAHSIC* fs=new CBAHSIC();
	index_t target_dim=dim/2;
	fs->set_labels(labels);
	fs->set_target_dim(target_dim);
	fs->set_kernel_features(kernel_p);
	fs->set_kernel_labels(kernel_q);
	fs->set_policy(N_SMALLEST);
	fs->set_num_remove(dim-target_dim);
	CFeatures* selected=fs->apply(feats);

	SGMatrix<float64_t> selected_data
		=((CDenseFeatures<float64_t>*)selected)->get_feature_matrix();

	// ensure that the selected number of features is indeed equal to the
	// target dimension
	EXPECT_EQ(selected_data.num_rows, target_dim);

	// ensure that selected feats are the same as computed in local machine
	SGVector<index_t> inds(target_dim);
	inds[0]=2;
	inds[1]=5;
	inds[2]=6;
	inds[3]=7;

	for (index_t i=0; i<target_dim; ++i)
	{
		for (index_t j=0; j<num_data; ++j)
			EXPECT_NEAR(data(inds[i], j), selected_data(i, j), 1E-15);
	}

	SG_UNREF(selected);
	SG_UNREF(fs);
	SG_UNREF(feats);
}
