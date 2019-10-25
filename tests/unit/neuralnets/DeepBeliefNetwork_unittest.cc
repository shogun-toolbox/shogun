/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2014 Khaled Nasr
 */
#include <gtest/gtest.h>
#include <shogun/neuralnets/DeepBeliefNetwork.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/UniformRealDistribution.h>

using namespace shogun;

TEST(DeepBeliefNetwork, convert_to_neural_network)
{
	const int32_t seed = 100;
	std::mt19937_64 prng(seed);

	DeepBeliefNetwork dbn(5, RBMVUT_BINARY);
	dbn.add_hidden_layer(6);
	dbn.add_hidden_layer(4);
	dbn.add_hidden_layer(8);

	dbn.initialize_neural_network();

	auto nn = dbn.convert_to_neural_network();

	UniformRealDistribution<float64_t> uniform_real_dist(0.0, 1.0);
	SGMatrix<float64_t> x(5, 3);
	for (int32_t i=0; i<x.num_rows*x.num_cols; i++)
		x[i] = uniform_real_dist(prng);

	auto f = std::make_shared<DenseFeatures<float64_t>>(x);

	auto f_transformed_dbn = dbn.transform(f);
	auto f_transformed_nn = nn->transform(f);

	SGMatrix<float64_t> x_transformed_dbn =
		f_transformed_dbn->get_feature_matrix();

	SGMatrix<float64_t> x_transformed_nn =
		f_transformed_nn->get_feature_matrix();

	for (int32_t i=0; i< x_transformed_dbn.num_rows*x_transformed_dbn.num_cols; i++)
		EXPECT_NEAR(x_transformed_dbn[i], x_transformed_nn[i], 1e-15);

}
