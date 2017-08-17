/*
 * Copyright (c) 2017, Shogun Toolbox Foundation
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
 * Written (W) 2017 Olivier Nguyen
 */

#include <shogun/neuralnets/NeuralRecurrentLayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

class NeuralRecurrentLayer: public ::testing::Test
{
public:
	virtual void SetUp()
	{
	}

	virtual void TearDown()
	{
	}

};


/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralRecurrentLayer, compute_activations)
{
}

/** Compares the error computed using the layer against a manually computed
 * error
 */
TEST_F(NeuralRecurrentLayer, compute_error)
{
}

/** Compares the local gradients computed using the layer against gradients
 * computed using numerical approximation
 */
TEST_F(NeuralRecurrentLayer, compute_local_gradients)
{
}

/** Compares the parameter gradients computed using the layer, when the layer
 * is used as an output layer, against gradients computed using numerical
 * approximation
 */
TEST_F(NeuralRecurrentLayer, compute_parameter_gradients_output)
{
}

/** Compares the parameter gradients computed using the layer, when the layer
 * is used as a hidden layer, against gradients computed using numerical
 * approximation
 */
TEST_F(NeuralRecurrentLayer, compute_parameter_gradients_hidden)
{
}
