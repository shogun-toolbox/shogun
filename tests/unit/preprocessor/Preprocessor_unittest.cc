/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/SortWordString.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

#include <random>

using namespace shogun;

TEST(Preprocessor, dense)
{
	const int32_t seed = 100;
	const index_t dim=2;
	const index_t size=4;

	std::mt19937_64 prng(seed);
	NormalDistribution<float64_t> normal_dist;
	SGMatrix<float64_t> data(dim, size);
	for (index_t i=0; i<dim*size; ++i)
		data.matrix[i]=normal_dist(prng);

	auto features = std::make_shared<DenseFeatures<float64_t>>(data);
	auto preproc = std::make_shared<NormOne>();
	preproc->fit(features);

	auto preprocessed = preproc->transform(features);

	EXPECT_EQ(preprocessed->get_feature_class(), C_DENSE);
}

TEST(Preprocessor, string)
{
	const int32_t seed = 100;
	const index_t num_strings=3;
	const index_t max_string_length=20;
	const index_t min_string_length=max_string_length/2;

	std::vector<SGVector<uint16_t>> strings;
	strings.reserve(num_strings);
	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	for (index_t i=0; i<num_strings; ++i)
	{
		index_t len=uniform_int_dist(prng, {min_string_length, max_string_length});
		SGVector<uint16_t> current(len);

		/* fill with random uppercase letters (ASCII) */
		random::fill_array(current, 'A', 'Z', prng);
		strings.push_back(current);
	}

	/* create num_features 2-dimensional vectors */
	auto features = std::make_shared<StringFeatures<uint16_t>>(strings, ALPHANUM);
	auto preproc = std::make_shared<SortWordString>();
	preproc->fit(features);

	auto preprocessed = preproc->transform(features);

	ASSERT_NE(preprocessed, nullptr);
	EXPECT_EQ(preprocessed->get_feature_class(), C_STRING);
}
