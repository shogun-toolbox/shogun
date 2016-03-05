/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2016 Youssef Emad El-Din
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
 *
 */
#include <shogun/lib/config.h>


#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>
#include <shogun/regression/Regression.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

TEST(LinearMachine,apply_train)
{
	index_t n=20;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, 3);
	SGVector<float64_t> Y(n);

	X_test[0]=3;
	X_test[1]=7.5;
	X_test[2]=12;

	for (index_t i=0; i<n; ++i)
	{
		X[i] = i;
		Y[i]=2*X[i] +1;
	}

	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	float64_t tau=0.8;
	CLinearRidgeRegression* model=new CLinearRidgeRegression(tau, feat_train,label_train);
	model->train();

	CRegressionLabels* predictions=model->apply_regression(feat_test);
	SGVector<float64_t> prediction_vector=predictions->get_labels();

	EXPECT_LE(CMath::abs(prediction_vector[0]-7), 0.5);
	EXPECT_LE(CMath::abs(prediction_vector[1]-16), 0.5);
	EXPECT_LE(CMath::abs(prediction_vector[2]-25), 0.5);
}

TEST(LinearMachine,compute_bias)
{
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;


	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	float64_t tau=0.8;
	CLinearRidgeRegression* model=new CLinearRidgeRegression(tau, feat_train,label_train);
	model->train();
	float64_t output_bias = model->get_bias();

	CLinearRidgeRegression* model_nobias=new CLinearRidgeRegression(tau, feat_train,label_train);
	model_nobias->set_compute_bias(false);
	model_nobias->train();

	// Calculate bias manually
	CRegressionLabels* predictions_nobias = model_nobias->apply_regression(feat_train);
	SGVector<float64_t> prediction_vector = predictions_nobias->get_labels();

	Map<VectorXd> eigen_prediction(prediction_vector.vector, 3);
	Map<VectorXd> eigen_labels(Y, 3);
	float64_t expected_bias = (eigen_labels-eigen_prediction).mean();

	EXPECT_LE(CMath::abs(output_bias - expected_bias), 10E-15);
}
