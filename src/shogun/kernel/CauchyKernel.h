/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser
 */



#ifndef CAUCHYKERNEL_H_
#define CAUCHYKERNEL_H_

#include <shogun/kernel/ShiftInvariantKernel.h>

namespace shogun
{

/** @brief Cauchy kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = \frac{1}{1+\frac{\| x-x' \|^2}{\sigma}}
 * \f]
 *
 */

class CauchyKernel: public ShiftInvariantKernel
{
public:
	/** default constructor */
	CauchyKernel();

	/** constructor
	 * @param cache size of cache
	 * @param sigma kernel parameter sigma
	 */
	CauchyKernel(int32_t cache, float64_t sigma);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param sigma kernel parameter sigma
	 */
	CauchyKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sigma);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	/**
	 * @return kernel type
	 */
	EKernelType get_kernel_type() override { return K_CAUCHY; }

	/**
	 * @return type of features
	 */
	EFeatureType get_feature_type() override { return m_distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	EFeatureClass get_feature_class() override { return m_distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	const char* get_name() const override { return "CauchyKernel"; }

	~CauchyKernel() override;

protected:

	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	float64_t compute(int32_t idx_a, int32_t idx_b) override;

private:

	void init();

protected:

	/// sigma parameter of kernel
	float64_t m_sigma;

};

}

#endif /* CAUCHYKERNEL_H_ */
