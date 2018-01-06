/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2018 Chinmay Kousik
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those
 * of the authors and should not be interpreted as representing official
 * policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/distributions/Gaussian.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

TEST(Gaussian, log_pdf_single_1d)
{
	SGVector<float64_t> mean(1), point(1);
	SGMatrix<float64_t> cov(1, 1);

	mean[0] = 0;
	point[0] = 0;
	cov(0,0) = 1.0;

	auto gauss = some<CGaussian>(mean, cov, FULL);
	float64_t log_pdf = gauss->compute_log_PDF(point);
	EXPECT_NEAR(log_pdf, -0.91893853320467267, 1e-16);
}

TEST(Gaussian, log_pdf_multiple_independent_2d)
{
	SGVector<float64_t> mean(2);
	SGMatrix<float64_t> cov(2, 2);

	mean[0] = mean[1] = 0;
	cov(0,0) = 1.0;
	cov(0,1) = 0.0;
	cov(1,0) = 0.0;
	cov(1,1) = 1.0;

	auto gauss = some<CGaussian>(mean, cov, FULL);
	float64_t log_pdf = gauss->compute_log_PDF(mean);
	EXPECT_NEAR(log_pdf, -1.8378770664093453, 1e-16);
}

TEST(Gaussian, log_pdf_multiple_2d)
{
	SGVector<float64_t> mean(2);
	SGMatrix<float64_t> cov(2,2);

	mean[0] = mean[1] = 0;

	cov(0,0) = 4.0;
	cov(0,1) = 1.0;
	cov(0,1) = 1.0;
	cov(1,1) = 2.0;
	auto gauss = some<CGaussian>(mean, cov, FULL);
	float64_t log_pdf = gauss->compute_log_PDF(mean);
	EXPECT_NEAR(log_pdf, -2.8108321409370021, 1e-16);
}

TEST(Gaussian, train_univariate)
{
	float64_t eps = 1e-8;
	sg_rand->set_seed(1);

	SGMatrix<float64_t> data(1, 500);

	int64_t sample_size = 500;
	float64_t mn = 0.0, cv = 0.0;

	for (int32_t i = 0; i < sample_size; i++)
		data(0, i) = CMath::randn_double();

	auto gauss = some<CGaussian>();
	auto train_features = some<CDenseFeatures<float64_t>>(data);
	gauss->train(train_features);

	auto mean = gauss->get_mean();
	auto cov = gauss->get_cov();

	for (int32_t i = 0; i < sample_size; i++)
		mn += data(0, i);
	mn *= 1.0 / sample_size;

	for (int32_t i = 0; i < sample_size; i++)
		cv += CMath::pow(data(0, i) - mn, 2);
	cv *= 1.0 / sample_size;

	EXPECT_NEAR(mean[0], mn, eps);
	EXPECT_NEAR(cov(0, 0), cv, eps);
}

TEST(Gaussian, train_bivariate)
{
	float64_t eps = 1e-8;
	sg_rand->set_seed(2);
	SGMatrix<float64_t> sample_cov(2, 2);
	SGVector<float64_t> sample_mean(2);

	int32_t sample_size = 500;

	sample_mean[0] = 0.0;
	sample_mean[1] = 2.0;

	sample_cov(0,0) = 2.0;
	sample_cov(1,1) = 4.0;
	sample_cov(0,1) = sample_cov(1,0) = 0.0;

	SGMatrix<float64_t> data(2, sample_size);

	for(int32_t i = 0; i < 2; i++){
		for (int32_t j = 0; j < sample_size; j++)
		{
			auto rd = CMath::randn_double();
			auto sigma = CMath::sqrt(sample_cov(i,i));
			data(i, j) = sigma*rd + sample_mean[i];
		}
	}

	auto gauss = some<CGaussian>();
	auto train_features = some<CDenseFeatures<float64_t>>(data);
	gauss->train(train_features);

	linalg::zero(sample_mean);
	linalg::zero(sample_cov);

	for (int32_t i = 0; i < sample_size; i++)
		train_features->add_to_dense_vec(1.0, i, sample_mean.vector, 2);
	linalg::scale(sample_mean, sample_mean, 1.0 / sample_size);

	for (int32_t i = 0; i < 2; i++)
	{
		for (int32_t j = 0; j < 2; j++)
		{
			for (int32_t k = 0; k < sample_size; k++)
			{
				float64_t x = data(i, k) - sample_mean[i];
				float64_t y = data(j, k) - sample_mean[j];
				sample_cov(i, j) += x * y;
			}
		}
	}
	linalg::scale(sample_cov, sample_cov, 1.0 / sample_size);

	auto cov = gauss->get_cov();
	auto mean = gauss->get_mean();

	for(int32_t i = 0; i < 2; i++){
		EXPECT_NEAR(mean[i], sample_mean[i], eps);
		for(int32_t j = 0; j < 2; j++){
			EXPECT_NEAR(cov(i, j), sample_cov(i, j), eps);
		}
	}
}
