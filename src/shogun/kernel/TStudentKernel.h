/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser
 */

#include <shogun/lib/config.h>

#ifndef TSTUDENTKERNEL_H_
#define TSTUDENTKERNEL_H_

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Distance;

/** @brief Generalized T-Student kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = \frac{1}{1+\| x-x' \|^degree}
 * \f]
 * with degree=1 by default
 */

class TStudentKernel: public Kernel
{
public:
	/** default constructor */
	TStudentKernel();

	/** constructor
	 * @param cache size of cache
	 * @param d kernel parameter degree
	 * @param dist distance to be used
	 */
	TStudentKernel(int32_t cache, float64_t d, std::shared_ptr<Distance> dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param d kernel parameter degree
	 * @param dist distance to be used
	 */
	TStudentKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t d, std::shared_ptr<Distance> dist);

	virtual ~TStudentKernel();

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_TSTUDENT; }

	/**
	 * @return type of features
	 */
	virtual EFeatureType get_feature_type() { return distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	virtual EFeatureClass get_feature_class() { return distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	virtual const char* get_name() const { return "TStudentKernel"; }

	/** getter for degree parameter
	 *  @return kernel parameter degree
	 */
	inline float64_t get_degree() { return this->degree; }

	/** setter for degree parameter
	 *  @param value kernel parameter degree
	 */
	inline void set_degree(float64_t value) { this->degree = value; }

private:
	void init();

protected:

	/// distance to be used
	std::shared_ptr<Distance> distance;

	/// degree parameter of kernel
	float64_t degree;

	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};
}

#endif /* TSTUDENTKERNEL_H_ */
