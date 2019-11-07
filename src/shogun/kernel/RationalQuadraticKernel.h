/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#ifndef RATIONAL_QUADRATIC_KERNEL_H_
#define RATIONAL_QUADRATIC_KERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Distance;

/** @brief Rational Quadratic kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = 1 -\frac{\| x-x'\|^2}{\|x-x'\|^2 + coef}
 * \f]
 *
 */

class RationalQuadraticKernel: public Kernel
{
public:
	/** default constructor */
	RationalQuadraticKernel();

	/** constructor
	 * @param cache size of cache
	 * @param coef kernel parameter coefficient
	 * @param dist distance to be used
	 */
	RationalQuadraticKernel(int32_t cache, float64_t coef, std::shared_ptr<Distance> dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param c kernel parameter coefficient
	 * @param dist distance to be used
	 */
	RationalQuadraticKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t c, std::shared_ptr<Distance> dist);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_RATIONAL_QUADRATIC; }

	/**
	 * @return type of features
	 */
	virtual EFeatureType get_feature_type() { return m_distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	virtual EFeatureClass get_feature_class() { return m_distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	virtual const char* get_name() const { return "RationalQuadraticKernel"; }

	/**
	 * @return coef - coefficient parameter of kernel
	 */
	virtual const float64_t get_coef() const { return m_coef; }

	/** setter for degree parameter
	 *  @param coef coefficient parameter of kernel
	 */
	inline void set_coef(float64_t coef) { m_coef=coef; }

	virtual ~RationalQuadraticKernel();
protected:
	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

private:
	/**Initialize parameters for serialization*/
	void init();

protected:
	/// distance to be used
	std::shared_ptr<Distance> m_distance;

	/// coefficient parameter of kernel
	float64_t m_coef;
};
}

#endif /* RATIONAL_QUADRATIC_KERNEL_H_ */
