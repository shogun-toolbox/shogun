/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/config.h>

#ifndef INVERSEMULTIQUADRIC_H_
#define INVERSEMULTIQUADRIC_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Distance;
/** @brief InverseMultiQuadricKernel
*
* \f[
*             K(x,x') = 1/(\sqrt{\| x - x' \|^2 +c^2})
* \f]
*/

class InverseMultiQuadricKernel: public Kernel
{
public:
	/** default constructor */
	InverseMultiQuadricKernel();

	/** constructor
	 * @param cache size of cache
	 * @param coef kernel parameter coef
	 * @param dist distance to be used
	 */
	InverseMultiQuadricKernel(int32_t cache, float64_t coef, std::shared_ptr<Distance> dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param coef kernel parameter coef
	 * @param dist distance to be used
	 */
	InverseMultiQuadricKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t coef, std::shared_ptr<Distance> dist);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

	/**
	 * @return kernel type
	 */
	EKernelType get_kernel_type() override { return K_INVERSEMULTIQUADRIC; }

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
	const char* get_name() const override { return "InverseMultiQuadricKernel"; }

	/** getter for coef parameter
	 *  @return kernel parameter coefficient
	 */
	inline float64_t get_coef() { return this->coef; }

	/** setter for coef parameter
	 *  @param value kernel parameter coefficient
	 */
	inline void set_coef(float64_t value) { this->coef = value; }

	~InverseMultiQuadricKernel() override;

	/** Can (optionally) be overridden to post-initialize some
	 *  member variables which are not PARAMETER::ADD'ed.  Make
	 *  sure that at first the overridden method
	 *  BASE_CLASS::LOAD_SERIALIZABLE_POST is called.
	 *
	 *  @exception ShogunException Will be thrown if an error
	 *                             occurres.
	 */
	void load_serializable_post() override;

protected:

	/** distance to be used
	 */
	std::shared_ptr<Distance> distance;

	/** theta parameter of kernel - coefficient
	 */
	float64_t coef;

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
};

}

#endif /* INVERSEMULTIQUADRIC_H_ */
