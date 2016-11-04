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
 
#include <shogun/neuralnets/RBM.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(RBM, gibbs_sampling)
{
	CMath::init_random(10);

	int32_t num_visible = 5;
	int32_t num_hidden = 6;

	CRBM rbm(num_hidden, num_visible, RBMVUT_BINARY);
	rbm.initialize_neural_network();
	rbm.set_batch_size(1);

	for (int32_t i=0; i<rbm.get_weights().num_rows*rbm.get_weights().num_cols; i++)
		rbm.get_weights()[i] = i*1.0e-2;

	for (int32_t i=0; i<rbm.get_hidden_bias().vlen; i++)
		rbm.get_hidden_bias()[i] = i*1.0e-1;

	for (int32_t i=0; i<rbm.get_visible_bias().vlen; i++)
		rbm.get_visible_bias()[i] = i*1.0e-1;

	for (int32_t i=0; i<num_visible; i++)
		rbm.visible_state[i] = i*1.0e-3;

	SGVector<float64_t> probs(num_visible);
	probs.zero();

	rbm.sample(1000);

	for (int32_t i=0; i<1000; i++)
	{
		rbm.sample(100);

		for (int32_t j=0; j<num_visible; j++)
			probs[j] += rbm.visible_state[j]/1000;
	}

	// generated using scikit-learn
	float64_t probs_ref[] = {0.5347, 0.6105, 0.6857, 0.7561, 0.8132};

	for (int32_t i=0; i<num_visible; i++)
		EXPECT_NEAR(probs_ref[i], probs[i], 0.02);
}

TEST(RBM, free_energy_binary)
{
	CMath::init_random(100);

	int32_t num_visible = 5;
	int32_t num_hidden = 6;
	int32_t batch_size = 3;

	CRBM rbm(num_hidden, num_visible, RBMVUT_BINARY);
	rbm.initialize_neural_network();

	for (int32_t i=0; i<rbm.get_weights().num_rows*rbm.get_weights().num_cols; i++)
		rbm.get_weights()[i] = i*1.0e-2;

	for (int32_t i=0; i<rbm.get_hidden_bias().vlen; i++)
		rbm.get_hidden_bias()[i] = i*1.0e-1;

	for (int32_t i=0; i<rbm.get_visible_bias().vlen; i++)
		rbm.get_visible_bias()[i] = i*1.0e-1;

	SGMatrix<float64_t> V(num_visible, batch_size);
	for (int32_t i=0; i<V.num_rows*V.num_cols; i++)
		V[i] = i*1e-3;

	// generated using scikit-learn
	EXPECT_NEAR(-5.0044376228, rbm.free_energy(V), 1e-6);
}

TEST(RBM, free_energy_gradients)
{
	CMath::init_random(100);

	int32_t num_visible = 15;
	int32_t num_hidden = 6;
	int32_t batch_size = 3;

	CRBM rbm(num_hidden);
	rbm.add_visible_group(4, RBMVUT_BINARY);
	rbm.add_visible_group(6, RBMVUT_GAUSSIAN);
	rbm.add_visible_group(5, RBMVUT_BINARY);
	rbm.initialize_neural_network();

	SGMatrix<float64_t> V(num_visible, batch_size);
	for (int32_t i=0; i<V.num_rows*V.num_cols; i++)
		V[i] = CMath::random() < 0.7;

	SGVector<float64_t> gradients(rbm.get_num_parameters());
	rbm.free_energy_gradients(V, gradients);

	SGVector<float64_t> params = rbm.get_parameters();
	SGVector<float64_t> gradients_numerical(rbm.get_num_parameters());
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<rbm.get_num_parameters(); i++)
	{
		params[i] += epsilon;
		float64_t energy_plus =rbm.free_energy(V);

		params[i] -= 2*epsilon;
		float64_t energy_minus =rbm.free_energy(V);

		params[i] += epsilon;

		gradients_numerical[i] = (energy_plus-energy_minus)/(2*epsilon);
	}

	for (int32_t i=0; i<gradients.vlen; i++)
		EXPECT_NEAR(gradients_numerical[i], gradients[i], 1e-6);
}

TEST(RBM, pseudo_likelihood_binary)
{
	CMath::init_random(100);

	int32_t num_visible = 5;
	int32_t num_hidden = 6;
	int32_t batch_size = 1;

	CRBM rbm(num_hidden, num_visible, RBMVUT_BINARY);
	rbm.initialize_neural_network();

	for (int32_t i=0; i<rbm.get_weights().num_rows*rbm.get_weights().num_cols; i++)
		rbm.get_weights()[i] = i*1.0e-2;

	for (int32_t i=0; i<rbm.get_hidden_bias().vlen; i++)
		rbm.get_hidden_bias()[i] = i*1.0e-1;

	for (int32_t i=0; i<rbm.get_visible_bias().vlen; i++)
		rbm.get_visible_bias()[i] = i*1.0e-1;

	SGMatrix<float64_t> V(num_visible, batch_size);
	for (int32_t i=0; i<V.num_rows*V.num_cols; i++)
		V[i] = i > 2;

	float64_t pl = 0;
	for (int32_t i=0; i<10000; i++)
		pl += rbm.pseudo_likelihood(V)/10000;

	// generated using scikit-learn
	EXPECT_NEAR(-3.3698, pl, 0.02);
}
