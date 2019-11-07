/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Heiko Strathmann
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2012 Jacob Walker
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
#include <shogun/machine/gp/LikelihoodModel.h>

#include <utility>

using namespace shogun;

LikelihoodModel::LikelihoodModel()
{
}

LikelihoodModel::~LikelihoodModel()
{
}

SGVector<float64_t> LikelihoodModel::get_predictive_log_probabilities(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels >lab)
{
	return get_log_zeroth_moments(mu, s2, std::move(lab));
}

SGVector<float64_t> LikelihoodModel::get_log_probability_fmatrix(
		std::shared_ptr<const Labels> lab, SGMatrix<float64_t> F) const
{
	require(lab, "Given labels are NULL!");
	require(lab->get_num_labels()==F.num_rows, "Number of labels ({}) does "
			"not match dimension of functions ({})",
			lab->get_num_labels(),F.num_rows);
	require(F.num_cols>0, "Number of passed functions ({}) must be positive",
			F.num_cols);

	SGVector<float64_t> result(F.num_cols);
	for (index_t i=0; i<F.num_cols; ++i)
	{
		/* extract current sample from matrix, assume col-major, dont copy */
		SGVector<float64_t> f(&F.matrix[i*F.num_rows], F.num_rows, false);
		result[i]=SGVector<float64_t>::sum(get_log_probability_f(lab, f));
	}

	return result;
}

SGVector<float64_t> LikelihoodModel::get_first_moments(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());

	SGVector<float64_t> result(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
		result[i]=get_first_moment(mu, s2, lab, i);

	return result;
}

SGVector<float64_t> LikelihoodModel::get_second_moments(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());

	SGVector<float64_t> result(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
		result[i]=get_second_moment(mu, s2, lab, i);

	return result;
}
