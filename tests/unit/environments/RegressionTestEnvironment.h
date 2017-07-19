/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 MikeLing, Viktor Gal, Sergey Lisitsyn, Heiko Strathmann
 */

#ifndef __REGRESSION_TEST_ENVIRONMENT_H
#define __REGRESSION_TEST_ENVIRONMENT_H

#include <gtest/gtest.h>
#include <memory>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;
using namespace std;
using ::testing::Environment;

class RegressionTestEnvironment : public ::testing::Environment
{
private:
	const index_t n_train = 20, n_test = 15, n_dim = 4;
	CDenseFeatures<float64_t> *features_train, *features_test;
	CRegressionLabels *labels_train, *labels_test;

public:
	virtual void SetUp()
	{
		set_global_seed(57);

		SGMatrix<float64_t> feat_train_data =
		    CDataGenerator::generate_gaussians(n_train, 1, n_dim);

		SGMatrix<float64_t> feat_test_data =
		    CDataGenerator::generate_gaussians(n_test, 1, n_dim);

		SGVector<float64_t> w(n_dim);
		w.random(-1, 1);

		SGVector<float64_t> label_train_data =
		    linalg::matrix_prod(feat_train_data, w, true);

		SGVector<float64_t> label_test_data =
		    linalg::matrix_prod(feat_test_data, w, true);

		features_train = new CDenseFeatures<float64_t>(feat_train_data);
		labels_train = new CRegressionLabels(label_train_data);

		features_test = new CDenseFeatures<float64_t>(feat_test_data);
		labels_test = new CRegressionLabels(label_test_data);

		SG_REF(features_train);
		SG_REF(labels_train);

		SG_REF(features_test);
		SG_REF(labels_test);
	}

	virtual void TearDown()
	{
		SG_UNREF(features_train);
		SG_UNREF(labels_train);

		SG_UNREF(features_test);
		SG_UNREF(labels_test);
	}

	CDenseFeatures<float64_t>* get_features_train() const
	{
		return features_train;
	}

	CDenseFeatures<float64_t>* get_features_test() const
	{
		return features_test;
	}

	CRegressionLabels* get_labels_train() const
	{
		return labels_train;
	}

	CRegressionLabels* get_labels_test() const
	{
		return labels_test;
	}
};

#endif
