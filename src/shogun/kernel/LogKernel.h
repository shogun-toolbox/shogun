/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser
 */

#include <shogun/lib/config.h>

#ifndef LOGKERNEL_H_
#define LOGKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Distance;

/** @brief Log kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = - log (\| x-x' \|^{degree} + 1)
 * \f]
 *
 */

class LogKernel: public Kernel
{
public:

	LogKernel();

	/** constructor
	 * @param cache size of cache
	 * @param degree kernel parameter degree
	 * @param dist distance to be used
	 */
	LogKernel(int32_t cache, float64_t degree, std::shared_ptr<Distance> dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param degree kernel parameter degree
	 * @param dist distance to be used
	 */
	LogKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t degree, std::shared_ptr<Distance> dist);

	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	EKernelType get_kernel_type() override { return K_POWER; }

	EFeatureType get_feature_type() override { return m_distance->get_feature_type(); }

	EFeatureClass get_feature_class() override { return m_distance->get_feature_class(); }

	const char* get_name() const override { return "LogKernel"; }

	~LogKernel() override;

protected:
	float64_t compute(int32_t idx_a, int32_t idx_b) override;

protected:
	std::shared_ptr<Distance> m_distance;
	float64_t m_degree = 1.8;
};
}

#endif /* LOGKERNEL_H_ */
