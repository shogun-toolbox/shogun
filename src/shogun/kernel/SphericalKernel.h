/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/lib/config.h>

#ifndef _SPHERICALKERNEL_H__
#define _SPHERICALKERNEL_H__

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Distance;

/** @brief Spherical kernel
 *
 * Formally described as
 *
 * \f[
 *     k(x, y) = 1 - \frac{3}{2} \frac{|x-y|}{\sigma}
 *     + \frac{1}{2} \left( \frac{|x-y|}{\sigma} \right)^3
 *     \mbox{if}~ |x-y| \leq \sigma \mbox{, zero otherwise}
 * \f]
 *
 */

class SphericalKernel: public Kernel
{
	public:
	/** default constructor */
	SphericalKernel();

	/** constructor
	 *
	 * @param size cache size
	 * @param sigma kernel parameter sigma
	 * @param dist distance
	 */
	SphericalKernel(int32_t size, float64_t sigma, std::shared_ptr<Distance> dist);

	/** constructor
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @param sigma kernel parameter sigma
	 * @param dist distance
	 */
	SphericalKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sigma, std::shared_ptr<Distance> dist);

	/** initialize kernel with features
	 *
	 * @param l features of left-side
	 * @param r features of right-side
	 * @return true if successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	/**
	 * @return kernel type
	 */
	EKernelType get_kernel_type() override { return K_SPHERICAL; }

	/**
	 * @return type of features
	 */
	EFeatureType get_feature_type() override { return distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	EFeatureClass get_feature_class() override { return distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	const char* get_name() const override { return "SphericalKernel"; }

	/** set the kernel's sigma
	 *
	 * @param s kernel sigma
	 */
	virtual void set_sigma(float64_t s)
	{
		sigma=s;
	}

	/** return the kernel's sigma
	 *
	 * @return kernel sigma
	 */
	virtual float64_t get_sigma() const
	{
		return sigma;
	}

	~SphericalKernel() override;

private:


protected:

	/** distance */
	std::shared_ptr<Distance> distance;

	/** width */
	float64_t sigma;

	/**
	 * compute kernel function for features a and b
	 * idx_{a,b} denote the index of the feature vectors
	 * in the corresponding feature object
	 *
	 * @param idx_a index a
	 * @param idx_b index b
	 * @return computed kernel function at indices a,b
	 */
	float64_t compute(int32_t idx_a, int32_t idx_b) override;
};
}

#endif /* _SPHERICALKERNEL_H__ */
