/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Evgeniy Andreev, Yuyu Zhang, 
 *          Bjoern Esser, Saurabh Goyal
 */

#ifndef _RANDOMFOURIERGAUSSPREPROC__H__
#define _RANDOMFOURIERGAUSSPREPROC__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/preprocessor/DensePreprocessor.h>

namespace shogun {
	/** @brief Preprocessor that approximates Gaussian feature map.
	 *
	 * Gaussian kernel of form \f$k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})\f$
	 * is approximated using the formula: 
	 *
	 * \f[
	 * 	z(x) = \sqrt{2/D}\ cos(wx + b)
	 *  k(x, y) \approx z'(x)z(y)
	 * \f]
	 * 
	 * Reference:
	 * [1] Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel
	 * machines. In Advances in neural information processing systems (pp.
	 * 1177-1184).
	 *
	 */
class RandomFourierGaussPreproc: public RandomMixin<DensePreprocessor<float64_t>> {
public:

	RandomFourierGaussPreproc();

	~RandomFourierGaussPreproc();

	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector) override;

	virtual EFeatureType get_feature_type() override
	{
		return F_DREAL;
	}

	virtual EFeatureClass get_feature_class() override
	{
		return C_DENSE;
	};

	virtual void fit(std::shared_ptr<Features> f) override;

	void cleanup();

	virtual const char* get_name() const override
	{
		return "RandomFourierGaussPreproc";
	}

	virtual EPreprocessorType get_type() const override
	{
		return P_RANDOMFOURIERGAUSS;
	}

	SG_FORCED_INLINE float64_t get_width() const
	{
		return std::exp(m_log_width * 2.0) * 2.0;
	}

	void set_width(float64_t width)
	{
		m_log_width = std::log(width/2.0) / 2.0;
	}

	SG_FORCED_INLINE int32_t get_dim_output() const
	{
		return m_dim_output;
	}

	void set_dim_output(int32_t dims)
	{
		m_dim_output = dims;
	}

	protected:

	virtual SGMatrix<float64_t> apply_to_matrix(SGMatrix<float64_t> matrix) override;

	virtual SGMatrix<float64_t> sample_spectral_density(int32_t dim_input_space) const;

	/** Helper method which generates random coefficients and stores in the
	 *  internal members.
	 *
	 * @param dim_input_space input space dimension
	 */
	virtual void init_basis(int32_t dim_input_space);

	float64_t m_log_width = -0.34657359027997264;

	int32_t m_dim_output = 100;

	/** Fourier basis offset */
	SGVector<float64_t> m_offset;

	/** Fourier basis coefficient */
	SGMatrix<float64_t> m_basis;
};
}
#endif
