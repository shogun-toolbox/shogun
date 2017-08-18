/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2017 Michele Mazzoni
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
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/iterators/DotIterator.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

std::pair<SGMatrix<float64_t>, SGVector<float64_t>> get_data()
{
	const index_t n_rows = 6, n_cols = 8;

	SGMatrix<float64_t> mat(n_rows, n_cols);
	for (index_t i = 0; i < n_rows * n_cols; ++i)
		mat[i] = CMath::randn_double();

	SGVector<float64_t> vec(n_rows);
	for (index_t i = 0; i < n_rows; ++i)
		vec[i] = CMath::randn_double();

	return std::make_pair(mat, vec);
}

TEST(DotIterator, dot)
{
	auto data = get_data();
	auto mat = data.first;
	auto vec = data.second;

	auto feats = some<CDenseFeatures<float64_t>>(mat);

	index_t i = 0;
	for (const auto& v : DotIterator(feats))
	{
		EXPECT_EQ(
			v.dot(vec),
			linalg::sum(linalg::element_prod(mat.get_column(i), vec))
		);
		++i;
	}
}

TEST(DotIterator, add)
{
	auto data = get_data();
	auto mat = data.first;
	auto alphas = data.second;

	auto vec = SGVector<float64_t>(mat.num_rows);
	auto res = SGVector<float64_t>(mat.num_rows);

	auto feats = some<CDenseFeatures<float64_t>>(mat);

	index_t i = 0;
	for (const auto& v : DotIterator(feats))
	{
		v.add(alphas[i], vec);
		linalg::add(res, mat.get_column(i), res, 1.0, alphas[i]);
		++i;
	}

	for (i = 0; i < vec.vlen; ++i)
		EXPECT_EQ(vec[i], res[i]);
}
