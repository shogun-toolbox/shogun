/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser
 */

#include <shogun/lib/config.h>

#ifndef WAVEKERNEL_H_
#define WAVEKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Distance;

/** @brief Wave kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = \frac{\theta}{\| x-x' \|} \sin \frac{\| x-x' \|}{\theta}
 * \f]
 *
 */

class WaveKernel: public Kernel
{
public:
	/** default constructor */
	WaveKernel();

	/** constructor
	 * @param cache size of cache
	 * @param theta kernel parameter theta
	 * @param dist distance to be used
	 */
	WaveKernel(int32_t cache, float64_t theta, std::shared_ptr<Distance> dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param theta kernel parameter theta
	 * @param dist distance to be used
	 */
	WaveKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t theta, std::shared_ptr<Distance> dist);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	/**
	 * @return kernel type
	 */
	EKernelType get_kernel_type() override { return K_WAVE; }

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
	const char* get_name() const override { return "WaveKernel"; }

	~WaveKernel() override;

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

	/// distance to be used
	std::shared_ptr<Distance> m_distance;

	/// theta parameter of kernel
	float64_t m_theta;
};
}

#endif /* WAVEKERNEL_H_ */
