/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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


#ifndef QUADRATIC_TIME_MMD_H_
#define QUADRATIC_TIME_MMD_H_

#include <memory>
#include <shogun/statistical_testing/MMD.h>

namespace shogun
{

template <typename> class SGVector;

class CQuadraticTimeMMD : public CMMD
{
	using operation=std::function<float32_t(SGMatrix<float32_t>)>;
public:
	CQuadraticTimeMMD();
	CQuadraticTimeMMD(CFeatures* samples_from_p, CFeatures* samples_from_q);

	virtual ~CQuadraticTimeMMD();

	virtual float64_t compute_statistic() override;
	virtual float64_t compute_variance() override;

	virtual SGVector<float64_t> sample_null() override;
	void spectrum_set_num_eigenvalues(index_t num_eigenvalues);

	virtual float64_t compute_p_value(float64_t statistic) override;
	virtual float64_t compute_threshold(float64_t alpha) override;

	void precompute_kernel_matrix(bool precompute);

	virtual const char* get_name() const;
private:
	struct Self;
	std::unique_ptr<Self> self;

	virtual const operation get_direct_estimation_method() const override;
	virtual const float64_t normalize_statistic(float64_t statistic) const override;
	virtual const float64_t normalize_variance(float64_t variance) const override;
	SGVector<float64_t> gamma_fit_null();
	SGVector<float64_t> spectrum_sample_null();
};

}
#endif // QUADRATIC_TIME_MMD_H_
